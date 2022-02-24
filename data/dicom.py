# coding=utf-8
import numpy as np
import glob
import sys
import pydicom as dicom
import SimpleITK as sitk
import cv2
import scipy
from scipy import ndimage
# ------------------------------------------------------------------------#
# Read and write Dicom files
# ------------------------------------------------------------------------#
def read_dcm_series(dcm_directory):
    # Read the original series.
    # First obtain the series file names using the image series reader.
    #
    # Params :
    #    data_directory   - The direct directory of dicom series
    # Return :
    #    info          - Tag information of the first dicom
    #    array - image matrix
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dcm_directory)
    if not series_IDs:
        print("ERROR: given directory \"" + dcm_directory + "\" does not contain a DICOM series.")
        sys.exit(1)

    series_reader = sitk.ImageSeriesReader()
    image_info_list = []
    image_array_list = []
    dcmimage_path_list = []
    for series_ID in series_IDs:
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dcm_directory, series_ID)
        if len(series_file_names)<=1:
            continue
        else:
            # series_file_names = sorted(series_file_names)
            series_reader.SetFileNames(series_file_names)
            # Configure the reader to load all of the DICOM tags (public+private):
            # By default tags are not loaded (saves time).
            # By default if tags are loaded, the private tags are not loaded.
            # We explicitly configure the reader to load tags, including the
            series_reader.MetaDataDictionaryArrayUpdateOn()
            series_reader.LoadPrivateTagsOn()
            image_info = series_reader.Execute()
            image_array = np.array(sitk.GetArrayFromImage(image_info), dtype='int16')
            image_info_list.append(image_info)
            image_array_list.append(image_array)
            dcmimage_path_list.append(list(series_file_names))
    return image_info_list, image_array_list, dcmimage_path_list


# fiter the dcm by the tags
def dcm_filter_by_tag(dcm_info_list, tag_dict, exclude_tag_dict={}):
    if isinstance(dcm_info_list, list):
        statuslist = []
        for dcm_info in dcm_info_list:
            status = False
            for tag_key in tag_dict.keys():
                if hasattr(dcm_info, tag_key):
                    if dcm_info[tag_key] == tag_dict[tag_key]:
                        status = True

            for tag_key in exclude_tag_dict.keys():
                if hasattr(dcm_info, tag_key):
                    if dcm_info[tag_key] == tag_dict[tag_key]:
                        status = False
            statuslist.append(status)

    else:
        statuslist = False
        for tag_key in tag_dict.keys():
            if hasattr(dcm_info_list, tag_key):
                if dcm_info_list[tag_key] == tag_dict[tag_key]:
                    statuslist = True

        for tag_key in exclude_tag_dict.keys():
            if hasattr(dcm_info_list, tag_key):
                if dcm_info_list[tag_key] == tag_dict[tag_key]:
                    statuslist = False

    return statuslist






def read_dcm_structure(dcm_directory):
    RS_dcm_path = glob.glob(dcm_directory + '/RS*')
    if len(RS_dcm_path) > 1:
        print(dcm_directory + ' have more than one RS dicom. Caution: The program will just process the first one!')
    # Configure the reader to load all of the DICOM tags (public+private):
    # By default tags are not loaded (saves time).
    # By default if tags are loaded, the private tags are not loaded.
    # We explicitly configure the reader to load tags, including the
    # private ones.
    info = dicom.read_file(RS_dcm_path[0], force=True)

    structure_name = []
    # Used for check GTV_names
    # 读取所有ROI names
    # info.StructureSetROISequence 是该case的所有ROI序列
    # info.StructureSetROISequence[j].ROIName 是该case的第j个ROI序列的ROI name
    for j in range(len(info.StructureSetROISequence)):
        structure_name.append(info.StructureSetROISequence[j].ROIName)
    return structure_name


def get_dose(rd_info, image_info):
    # dose information
    dose_array = rd_info.pixel_array
    dose_pixelspace = rd_info.PixelSpacing
    dose_origin = rd_info.ImagePositionPatient
    DoseGridScaling = rd_info.DoseGridScaling
    # dose_array = np.array(dose_array, dtype='uint8')
    # image information
    image_pixelspace = image_info.GetSpacing()
    image_origin = image_info.GetOrigin()
    image_orient = image_info.GetDirection()
    # get new dose array
    converted_origin = np.dot(np.array(image_origin), np.array(np.reshape(image_orient, [3, 3])))
    direction = [-1 if converted_origin[i] / image_origin[i] < 0 else 1 for i in range(len(image_origin))]
    offset = [int((dose_origin[0] - image_origin[0])/image_pixelspace[0]*direction[0]),
              int((dose_origin[1] - image_origin[1])/image_pixelspace[1]*direction[1]),
              int((dose_origin[2] - image_origin[2])/image_pixelspace[2]*direction[2])]

    print(offset)
    image_array = np.array(sitk.GetArrayFromImage(image_info), dtype='int16')
    dose = np.zeros(np.shape(image_array), dtype='float32')
    scale = dose_pixelspace[0]/image_pixelspace[0]
    for z in range(offset[2], dose_array.shape[0]):
        temp = scipy.ndimage.interpolation.zoom(dose_array[z, :, :], [scale, scale], order=3, mode='nearest', prefilter=False)
        dose[z, offset[1]:offset[1]+temp.shape[0], offset[0]:offset[0]+temp.shape[1]] = temp
    dose = np.array(dose * DoseGridScaling * 100, dtype='float32')  # unit = cGY
    return dose


def cal_DVH(dose, mask):
    index = np.where(mask == 1)
    dose_list = []
    for i in range(len(index[0])):
        dose_list.append(dose[index[0][i], index[1][i], index[2][i]])
    volume = np.sum(mask)
    dose_list = np.sort(dose_list)
    dose_list = dose_list[::-1]
    volume_space = range(0, 101)
    DVH = {}
    for index in volume_space:
        m = float(index)/100.0*volume
        m_ = int(np.floor(m))
        alpha = m - m_
        try:
            if m_ == 0:
                DVH['D{}'.format(index)] = (1-alpha) * dose_list[0]
            else:
                if m_ == len(dose_list):
                    DVH['D{}'.format(index)] = dose_list[-1]
                else:
                    DVH['D{}'.format(index)] = alpha*dose_list[m_-1] + (1-alpha)*dose_list[m_]
        except:
            print("************{}************{}**************{}*************{}".format(m, m_, alpha, volume))
    DVH['D100'] = dose_list[-1]
    DVH['Dmin'] = np.min(dose_list)
    DVH['Dmax'] = np.max(dose_list)
    DVH['Dmean'] = np.mean(dose_list)
    return DVH

def read_struct(dcm_directory, ST_list, data_info):
    '''
    Read RT Struct for a series and transform the data into the pixel space
    Params
        dcm_directory/dcm_path   - The directory or path of dicom struction file.
        data_info  - Data info by SimpleITK.
    Return
        ST_mask    - key:  ST_name
                   - ST_mask[key]:  mask_array of structure key
    '''
    if '.dcm' in dcm_directory and isinstance(dcm_directory, str):
        info = dicom.read_file(dcm_directory, force=True)
    else:
        if isinstance(dcm_directory, list) and len(dcm_directory):
            dcm_directory = dcm_directory[0]# if there are multiple RS file, just process the first one
        else:
            print("No RS path or directory got" )
            return -1
        if 'RS' in dcm_directory or 'RT' in dcm_directory and '.dcm' in dcm_directory:
            RS_dcm_path = dcm_directory
            info = dicom.read_file(RS_dcm_path, force=True)
        else:
            RS_dcm_path = glob.glob(dcm_directory + '/RS*')
            if len(RS_dcm_path) > 1:
                print(dcm_directory + ' have more than one RS dicom. Caution: The program will just process the first one!')
            info = dicom.read_file(RS_dcm_path[0], force=True)

    # if len(info.StructureSetROISequence)==1:
    #     ST_Data, ST_zpos_range = set_ST_data(info, 0) #for what
    #     ST_slicenum = len(info.ROIContourSequence[0].ContourSequence)#for what?
    # else:
    if True:
        ST_mask = {}
        c=0
        name_do ={}
        for ROI_i in range(len(info.StructureSetROISequence)):
            for ST_name in ST_list:
                if ST_name == info.StructureSetROISequence[ROI_i].ROIName:
                # if ST_name in info.StructureSetROISequence[ROI_i].ROIName.lower() and 'AI' in info.StructureSetROISequence[ROI_i].ROIName:
                    # print(ST_name)
                    try:
                        ST_slicenum = len(info.ROIContourSequence[ROI_i].ContourSequence)
                    except:
                        print(ST_name + ' has no contour!')
                        continue
                    if ST_slicenum == 0:
                        continue
                    c=c+1
                    ST_Data, ST_zpos_range = set_ST_data(info, ROI_i)

                    # Get the origin and pixelspace
                    try:
                        left_up = data_info.GetOrigin()
                        pixelspace = data_info.GetSpacing()
                        orientation = data_info.GetDirection()
                    except:
                        left_up = data_info['origin']
                        pixelspace = data_info['pixelspacing']
                        orientation = data_info['orientation']

                    ST_pixData = ST_Data  # pass by value
                    for i in range(ST_slicenum):
                        length = ST_pixData[i].shape[0]
                        # physical position to pixel index
                        for j in range(length):
                            # ST_pixData[i][j][0] = np.around((ST_pixData[i][j][0]-left_up[0])/pixelspace[0]*orientation[0])
                            # ST_pixData[i][j][1] = np.around((ST_pixData[i][j][1]-left_up[1])/pixelspace[1]*orientation[4])
                            # ST_pixData[i][j][2] = np.around((ST_pixData[i][j][2]-left_up[2])/pixelspace[2]*orientation[8])
                            ST_pixData[i][j][0] = np.around(
                                (ST_pixData[i][j][0] - left_up[0]) / pixelspace[0] * orientation[0])
                            ST_pixData[i][j][1] = np.around(
                                (ST_pixData[i][j][1] - left_up[1]) / pixelspace[1] * orientation[4])
                            ST_pixData[i][j][2] = np.around(
                                (ST_pixData[i][j][2] - left_up[2]) / pixelspace[2] * orientation[8])

                        # Linear interpolation
                        # not suitable for contous with same points in all slices
                        # ST_pixData[i] = np.array(IntensifyContour(ST_pixData[i]))
                    # the modified approach
                    ST_pixData = np.array(list(np.array(IntensifyContour(ST_Data[i])) for i in range(len(ST_pixData))))
                    if ST_pixData is not None:
                        ST_array = contour2mask(ST_pixData, data_info)
                        ST_mask[info.StructureSetROISequence[ROI_i].ROIName] = np.array(ST_array, dtype='int16')
                        # ST_mask[ST_name] = np.array(ST_array, dtype='int16')
                        # ST_mask[ST_name] = np.transpose(np.array(ST_array, dtype='int16'),[0, 2, 1])
                    else:
                        print("There is no target structure in the ST of "+dcm_directory+'.')
        if c>43:
            print("Warning: there are {} OARs repeated!".format(c-43))
        if ST_mask is not None:
            return ST_mask
        else:
            print("Error in Reading RT from" + dcm_directory + '.')
            return -1

def read_structwithcolor(dcm_directory, ST_list, data_info):
    '''
    Read RT Struct for a series and transform the data into the pixel space
    Params
        dcm_directory/dcm_path   - The directory or path of dicom struction file.
        data_info  - Data info by SimpleITK.
    Return
        ST_mask    - key:  ST_name
                   - ST_mask[key]:  mask_array of structure key
    '''
    if isinstance(dcm_directory, list) and len(dcm_directory):
        dcm_directory = dcm_directory[0]# if there are multiple RS file, just process the first one
    else:
        print("No RS path or directory got" )
        return -1
    if 'RS' in dcm_directory and '.dcm' in dcm_directory:
        RS_dcm_path = dcm_directory
        info = dicom.read_file(RS_dcm_path, force=True)
    else:
        RS_dcm_path = glob.glob(dcm_directory + '/RS*')
        if len(RS_dcm_path) > 1:
            print(dcm_directory + ' have more than one RS dicom. Caution: The program will just process the first one!')
        info = dicom.read_file(RS_dcm_path[0], force=True)

    if len(info.StructureSetROISequence)==1:
        ST_Data, ST_zpos_range = set_ST_data(info, 0) #for what
        ST_slicenum = len(info.ROIContourSequence[0].ContourSequence)#for what?
    else:
        ST_mask = {}
        ST_color = {}
        c=0
        name_do ={}
        for ROI_i in range(len(info.StructureSetROISequence)):
            for ST_name in ST_list:
                if ST_name == info.StructureSetROISequence[ROI_i].ROIName:
                    # print(ST_name)
                    try:
                        ST_slicenum = len(info.ROIContourSequence[ROI_i].ContourSequence)
                    except:
                        print(ST_name + ' has no contour!')
                        continue
                    c=c+1
                    ST_Data, ST_zpos_range = set_ST_data(info, ROI_i)

                    # Get the origin and pixelspace
                    left_up = data_info.GetOrigin()
                    pixelspace = data_info.GetSpacing()
                    orientation = data_info.GetDirection()
                    thickness = np.around(data_info.GetSpacing()[2])
                    bottom = np.around(data_info.GetOrigin()[2] / thickness)

                    ST_pixData = ST_Data  # pass by value
                    for i in range(ST_slicenum):
                        length = ST_pixData[i].shape[0]
                        # physical position to pixel index
                        for j in range(length):
                            ST_pixData[i][j][0] = np.around((ST_pixData[i][j][0]-left_up[0])/pixelspace[0]*orientation[0])
                            ST_pixData[i][j][1] = np.around((ST_pixData[i][j][1]-left_up[1])/pixelspace[1]*orientation[4])
                            ST_pixData[i][j][2] = np.around((ST_pixData[i][j][2]-left_up[2])/pixelspace[2]*orientation[8])

                        # Linear interpolation
                        # not suitable for contous with same points in all slices
                        # ST_pixData[i] = np.array(IntensifyContour(ST_pixData[i]))
                    # the modified approach
                    ST_pixData = np.array(list(np.array(IntensifyContour(ST_Data[i])) for i in range(len(ST_pixData))))
                    if ST_pixData is not None:
                        ST_array = contour2mask(ST_pixData, data_info)
                        ST_mask[ST_name] = np.transpose(np.array(ST_array, dtype='int16'),[0, 2, 1])
                        ST_color[ST_name] = info.ROIContourSequence[ROI_i].ROIDisplayColor
                    else:
                        print("There is no target structure in the ST of "+dcm_directory+'.')
        if c>43:
            print("Warning: there are {} OARs repeated!".format(c-43))
        if ST_mask is not None:
            return ST_mask, ST_color
        else:
            print("Error in Reading RT from" + dcm_directory + '.')
            return -1

def set_ST_data(info, j):
    '''
    Get ROI array and the range of z given information of one ROI sequence
    Params:
        info          - Dicom information.
        j             - The j-th ROI sequence.
    Return:
        ST_Data       - ROI array.
        ST_zpos_range - The range of z-axis.
    '''
    ST_slicenum = len(info.ROIContourSequence[j].ContourSequence)
    contourData = []
    tempZlist = []

    for i in range(ST_slicenum):
        contourData.append(np.reshape(info.ROIContourSequence[j]. \
                                      ContourSequence[i].ContourData, (-1, 3)))
        tempZlist.append(contourData[i][0, 2])
    ST_zpos_range = [min(tempZlist), max(tempZlist)]
    ST_Data = np.array(contourData)
    return ST_Data, ST_zpos_range

def IntensifyContour(contour):
    """
    Linear interpolate when the distance of two point is far more than sqrt(2),
    other point need be added in in case imfill leak.
    Require:
        contour - 2D numpy array (n*3) of points (x,y,z).
    Output:
        newList - New array after interpretation.
    """
    length = contour.shape
    length = length[0]
    zpos = contour[0,2]
    newList = []
    newList.append([contour[0, 0], contour[0, 1], zpos])
    for i in range(length-1):
        cx = contour[i, 0]; cy = contour[i, 1]
        nextx = contour[i+1, 0]; nexty = contour[i+1, 1]
        if cx == nextx and cy == nexty:
            continue
        elif abs(cx-nextx)>1.1 or abs(cy-nexty)>1.1:
            dis1 = abs(cx-nextx); dis2 = abs(cy-nexty)
            dis1 = int(max(dis1, dis2))
            vec1 = np.linspace(cx, nextx, dis1+1).astype(int)
            # vec1 = np.linspace(cx, nextx, dis1+1)
            vec2 = np.linspace(cy, nexty, dis1+1).astype(int)
            # vec2 = np.linspace(cy, nexty, dis1+1)
            for j in range(len(vec1)):
                newList.append([vec1[j], vec2[j], zpos])
        else:
            newList.append([cx, cy, zpos])
    cx = contour[length-1, 0]; cy = contour[length-1, 1]
    nextx = contour[0, 0]; nexty = contour[0, 1]

    if cx == nextx and cy == nexty:
        return newList
    elif abs(cx - nextx) > 1.1 or abs(cy - nexty) > 1.1:
        dis1 = abs(cx - nextx)
        dis2 = abs(cy - nexty)
        dis1 = int(max(dis1, dis2))
        vec1 = np.linspace(cx, nextx, dis1 + 1).astype(int)
        vec2 = np.linspace(cy, nexty, dis1 + 1).astype(int)
        for j in range(len(vec1)):
            newList.append([vec1[j], vec2[j], zpos])
    else:
        newList.append([cx, cy, zpos])
    return newList

# -----------------------------------------------------------------------------
# transfer contour to 3D mask, full filling the internal voxels with 1
# -----------------------------------------------------------------------------
def Imgfill(rect):
    """
    Region growing method
    Args:
        rect - A numpy array whose points on
               boundary are 1 and the rest are 0
    Return:
        mask - A binary mask: points inside the contour label are 1.
    """
    w, h = rect.shape
    mask = rect.copy()
    # (0,0) (0,-1) seed point and label is 2
    mask[0][0] = 2; mask[0][h-1] = 2
    mask[w - 1][h - 1] = 2; mask[w - 1][0] = 2
    xx = [0, 0, w - 1, w - 1]
    yy = [0, h - 1, h - 1, 0]
    while True:
        new_xx = []
        new_yy = []
        for i in range(len(xx)):
            x = xx[i]; y = yy[i]
            if (x-1) > -1:
                if mask[x-1][y] == 0:
                    mask[x-1][y] = 2
                    new_xx.append(x-1)
                    new_yy.append(y)

            if (x+1) < w:
                if mask[x+1][y] == 0:
                    mask[x+1][y] = 2
                    new_xx.append(x+1)
                    new_yy.append(y)

            if (y-1) > -1:
                if mask[x][y-1] == 0:
                    mask[x][y-1] = 2
                    new_xx.append(x)
                    new_yy.append(y-1)

            if (y+1) < h:
                if mask[x][y+1] == 0:
                    mask[x][y+1] = 2
                    new_xx.append(x)
                    new_yy.append(y+1)
            # print("unfinished!")
        if len(new_yy) == 0:
            break
        xx = new_xx; yy = new_yy
    # change label value
    mask[np.where(mask == 0)] = 1
    mask[np.where(mask == 2)] = 0
    return mask

def contour2mask(ST_data, img_info):
    ## get the image geometric parameters
    # img3D_spacing  = img_info.GetSpacing()
    # img3D_origin = img_info.GetOrigin()
    try:
        img3D_size = img_info.GetSize()
    except:
        img3D_size = img_info['imagesize']

    # get the physical coordinates along the z axis

    # get the 3D binary image of contour, keep consist with the zyx format(Dicom)
    ST_mask = np.zeros((img3D_size[2], img3D_size[1]+4, img3D_size[0]+4))
    for ST_num in range(len(ST_data)):
        wrong_contour = 0
        ST_array= np.zeros((img3D_size[1], img3D_size[0]))
        slice = ST_data[ST_num][0][2].astype(int)
        for ST_z in range(len(ST_data[ST_num])):
            ST_xx = ST_data[ST_num][ST_z][0].astype(int)
            ST_yy = ST_data[ST_num][ST_z][1].astype(int)
            try:
                ST_array[ST_yy][ST_xx] = 1
            except:
                wrong_contour = 1
        ST_array_ = np.zeros((img3D_size[1]+4, img3D_size[0]+4))
        ST_array_[2:-2, 2:-2] = ST_array
        index_X, index_Y = np.where(ST_array_ == 1)
        if len(index_X)>0 and wrong_contour!=1:
            temp_ROI = ST_array_[index_X.min()-2:index_X.max()+2, index_Y.min()-2:index_Y.max()+2]
            ST_mask[slice, index_X.min() - 2:index_X.max() + 2, index_Y.min() - 2:index_Y.max() + 2] += Imgfill(temp_ROI)
    ST_mask[ST_mask>1]=0
    ST_mask = ST_mask[:, 2:-2, 2:-2]
    return ST_mask

def resample_volume(img_array, Original_spcing, Target_spacing):
    """
    Target_spacing: the voxel space in x, y and z directions

    """
    img3D_size = img_array.shape
    img3D_spacing = Original_spcing

    # image resize to target resolution
    physical_z = img3D_size[0] * img3D_spacing[0]
    physical_y = img3D_size[1] * img3D_spacing[1]
    physical_x = img3D_size[2] * img3D_spacing[2]
    respace_z = np.round(physical_z/Target_spacing[0]).astype(int)
    respace_y = np.round(physical_y/Target_spacing[1]).astype(int)
    respace_x = np.round(physical_x/Target_spacing[2]).astype(int)
    import matplotlib.pyplot as plt
    if Target_spacing[0]==img3D_spacing[0]:
        Resampled_img = np.zeros((respace_z, respace_y, respace_x))
        # resize the image slice by slice
        for size_z in range(img3D_size[0]):
            temp_mid_img = img_array[size_z, :, :]
            temp_mid_img = cv2.resize(temp_mid_img, (respace_y, respace_x), interpolation=cv2.INTER_NEAREST)
            # plt.imshow(temp_mid_img)
            # plt.show()
            Resampled_img[size_z, :, :] = temp_mid_img
    else:
        # resize 3D volume via scipy
        resize_factor = np.array(img3D_spacing)/np.array(Target_spacing)
        Resampled_img = ndimage.interpolation.zoom(img_array, resize_factor, order=3, prefilter=False)
    return Resampled_img



def crop_or_pad_volume(volume, center, target_shape, padding_value=0):
    input_shape = volume.shape
    output_volume = np.ones(target_shape, dtype=volume.dtype) * padding_value
    fore_half_size = [int(a/2) for a in target_shape]
    back_half_size = np.subtract(target_shape, fore_half_size)
    input_sub_index = np.maximum(np.subtract(center, fore_half_size), [0, 0, 0])
    input_up_index = np.minimum(np.add(center, back_half_size), input_shape)
    input_sub_index = [int(a) for a in input_sub_index]
    input_up_index = [int(a) for a in input_up_index]
    output_sub_index = np.maximum(np.subtract(fore_half_size, np.subtract(center, input_sub_index)), [0, 0, 0])
    output_up_index = np.minimum(np.add(fore_half_size, np.subtract(input_up_index, center)), target_shape)
    output_sub_index = [int(a) for a in output_sub_index]
    output_up_index = [int(a) for a in output_up_index]
    try:
        output_volume[output_sub_index[0]:output_up_index[0], output_sub_index[1]:output_up_index[1],
        output_sub_index[2]:output_up_index[2]] = volume[input_sub_index[0]:input_up_index[0],
                                                  input_sub_index[1]:input_up_index[1],
                                                  input_sub_index[2]:input_up_index[2]]
        crop_position = np.array([[input_sub_index[0], input_up_index[0]], [input_sub_index[1], input_up_index[1]],
                                  [input_sub_index[2], input_up_index[2]]], dtype='int16')
        pad_position = np.array([[output_sub_index[0], output_up_index[0]], [output_sub_index[1], output_up_index[1]],
                                 [output_sub_index[2], output_up_index[2]]], dtype='int16')
    except:
        print('cropping or padding faild!')
        exit(-1)
    return output_volume, crop_position, pad_position



def crop_volume(volume, center, target_shape):
    input_shape = volume.shape
    fore_half_size = [int(a/2) for a in target_shape]
    back_half_size = np.subtract(target_shape, fore_half_size)
    sub_index = np.maximum(np.subtract(center, fore_half_size), [0, 0, 0])
    up_index = np.minimum(np.add(center, back_half_size), input_shape)
    sub_index = [int(a) for a in sub_index]
    up_index = [int(a) for a in up_index]
    output_volume = volume[sub_index[0]:up_index[0],sub_index[1]:up_index[1],sub_index[2]:up_index[2]]
    crop_position = np.array([[sub_index[0], up_index[0]],[sub_index[1], up_index[1]],[sub_index[2],up_index[2]]], dtype='int16')
    return output_volume,crop_position