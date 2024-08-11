import math 

def _calculate_distance(point_1_tuple,point_2_tuple):
    return math.sqrt((point_1_tuple[0]-point_2_tuple[0])*(point_1_tuple[0]-point_2_tuple[0])+(point_1_tuple[1]-point_2_tuple[1])*(point_1_tuple[1]-point_2_tuple[1]))

def _table_calculate_distance(row):
    distance = _calculate_distance((row['user_gaze_aoi_center_tuple_x'],row['user_gaze_aoi_center_tuple_y']),(row['agent_gaze_aoi_center_tuple_x'],row['agent_gaze_aoi_center_tuple_y']))
    return distance

def _find_aoi_center_x_ratio(row,aoi_info_table,type_name):
    slide_id_from_zero = row['slide_id']
    course_name = row['course_name']
    transcript_id = row['transcript_id']
    aoi_id = int(row[type_name])

    aoi_item_available = aoi_info_table[(aoi_info_table['slide_id_from_zero']==slide_id_from_zero)&(aoi_info_table['course_name']==course_name)&(aoi_info_table['transcript_id']==transcript_id)]
    aoi_id_list = list(set(aoi_item_available['aoi_id']))
    aoi_item_related = aoi_info_table[(aoi_info_table['slide_id_from_zero']==slide_id_from_zero)&(aoi_info_table['course_name']==course_name)&(aoi_info_table['transcript_id']==transcript_id)&(aoi_info_table['aoi_id']==aoi_id)]
    try:
        aoi_center_x = aoi_item_related['aoi_center_x'].values[0]
        return aoi_center_x
    except:
        aoi_id_except = min(aoi_id_list) if aoi_id <= 0 else max(aoi_id_list)
        aoi_item_except = aoi_item_available[aoi_item_available['aoi_id']==aoi_id_except]
        aoi_center_x = aoi_item_except['aoi_center_x'].values[0]
        return aoi_center_x

    

def _find_aoi_center_y_ratio(row,aoi_info_table,type_name):
    slide_id_from_zero = row['slide_id']
    course_name = row['course_name']
    transcript_id = row['transcript_id']
    aoi_id = int(row[type_name])

    aoi_item_available = aoi_info_table[(aoi_info_table['slide_id_from_zero']==slide_id_from_zero)&(aoi_info_table['course_name']==course_name)&(aoi_info_table['transcript_id']==transcript_id)]
    aoi_id_list = list(set(aoi_item_available['aoi_id']))
    aoi_item_related = aoi_info_table[(aoi_info_table['slide_id_from_zero']==slide_id_from_zero)&(aoi_info_table['course_name']==course_name)&(aoi_info_table['transcript_id']==transcript_id)&(aoi_info_table['aoi_id']==aoi_id)]

    try:
        aoi_center_y = aoi_item_related['aoi_center_y'].values[0]
        return aoi_center_y
    except:
        aoi_id_except = min(aoi_id_list) if aoi_id <= 0 else max(aoi_id_list)
        aoi_item_except = aoi_item_available[aoi_item_available['aoi_id']==aoi_id_except]
        aoi_center_y = aoi_item_except['aoi_center_y'].values[0]
        return aoi_center_y
