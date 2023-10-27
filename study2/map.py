import numpy as np 
import pandas as pd 
import seaborn as sns
import geopandas as gpd
from matplotlib import pyplot as plt 
import openai
import os,sys,uuid,time,re,json,math 
import plotly.express as px


def _get_value_from_str(pupil_string):
    pupil_item = 0
    pupil_splits = pupil_string.split('-')

    for pupil_each in pupil_splits:
        pupil_item += float(pupil_each)
    return pupil_item/len(pupil_splits)



def draw_map_result(result_path,LAD_csv,geo_json,datatype):
    assert datatype in ['student_num','imd_band','avg_assessment','label_score','predict_score']
    # dataset UK info source: https://public.opendatasoft.com/explore/dataset/georef-united-kingdom-local-authority-district/table/?flg=en-us&disjunctive.ctry_code&disjunctive.ctry_name&disjunctive.rgn_code&disjunctive.rgn_name&disjunctive.ctyua_code&disjunctive.ctyua_name&disjunctive.lad_code&disjunctive.lad_name
    LAD_info = pd.read_csv(LAD_csv,sep='\t')
    region_list = list(set(LAD_info['Official Name Region']))

    region_LAD_dict = {}
    for region in region_list:
        LAD_info_item = LAD_info[LAD_info['Official Name Region']==region]
        region_LAD_dict[region] = list(set(LAD_info_item['Official Name Local authority district']))

    region_map = {'London Region':region_LAD_dict['London'], 'South Region':region_LAD_dict['South West']+region_LAD_dict['South East'], 'North Region':region_LAD_dict['North West']+region_LAD_dict['North East'], 'West Midlands Region':region_LAD_dict['West Midlands'], 'Yorkshire Region':region_LAD_dict['Yorkshire and The Humber'], 'East Midlands Region':region_LAD_dict['East Midlands'], 'Scotland':region_LAD_dict['Scotland'], 'South East Region':region_LAD_dict['South East'], 'Wales':region_LAD_dict['Wales'], 'Ireland':region_LAD_dict['Northern Ireland'], 'South West Region':region_LAD_dict['South West'], 'North Western Region':region_LAD_dict['North West'], 'East Anglian Region':region_LAD_dict['East of England']}
    student_num_dict = {}
    student_imd_dict = {}
    student_avg_assess_dict = {}
    student_real_score_dict = {}
    student_predict_score_dict = {}
    student_data = pd.read_csv(result_path)
    student_data = student_data.dropna()
    student_data_arr = np.array(student_data)
    student_id_list = list(set(student_data['student_id']))
    for i,student_id in enumerate(student_id_list):
        student_item = student_data[student_data['student_id']==student_id]
        region = student_item['region'].values[0]
        student_item_label = student_item[student_item['score_type']=='label_score']
        student_item_predict = student_item[student_item['score_type']=='predict_score_both']
        LAD_name_list = region_map[region]
        for LAD_name in LAD_name_list:
            if LAD_name in list(student_num_dict.keys()):
                assess_item = _get_value_from_str(student_item_label['past_scores'].values[0])
                if math.isnan(assess_item): continue
                student_num_dict[LAD_name] += 1
                student_imd_dict[LAD_name].append(student_item_label['imd_band'].values[0])
                student_avg_assess_dict[LAD_name].append(assess_item)
                student_real_score_dict[LAD_name].append(student_item_label['score_item'].values[0])
                student_predict_score_dict[LAD_name].append(student_item_predict['score_item'].values[0])
            else:
                student_num_dict[LAD_name] = 0
                student_imd_dict[LAD_name] = []
                student_avg_assess_dict[LAD_name] = []
                student_real_score_dict[LAD_name] = []
                student_predict_score_dict[LAD_name] = []

    # The LAD.json file comes from: https://raw.githubusercontent.com/thomasvalentine/Choropleth/main/Local_Authority_Districts_(December_2021)_GB_BFC.json
    # The code is adapted from: https://stackoverflow.com/questions/72972299/how-to-plot-a-plotly-choropleth-map-with-english-local-authorities-using-geojson
    with open(geo_json) as response:
        Local_authorities = json.load(response)

    la_data = []
    area_list = []
    val_list = []
    for i in range(len(Local_authorities["features"])):
        la = Local_authorities["features"][i]['properties']['LAD21NM']
        Local_authorities["features"][i]['id'] = la
        if la not in list(student_num_dict.keys()):
            continue
        else:
            # datatype in ['student_num','imd_band','avg_assessment','label_score','predict_score']
            if datatype == 'student_num':
                la_data.append([la,student_num_dict[la]])
            elif datatype == 'imd_band':
                la_data.append([la,np.mean(student_imd_dict[la])])
            elif datatype == 'avg_assessment':
                la_data.append([la,np.mean(student_avg_assess_dict[la])])
            elif datatype == 'label_score':    
                la_data.append([la,np.mean(student_real_score_dict[la])])
            elif datatype == 'predict_score':
                la_data.append([la,np.mean(student_predict_score_dict[la])])
        area_list.append(la)
    df = pd.DataFrame(la_data)
    df.columns = ['LAD','Value']

    fig = px.choropleth_mapbox(df,
                           geojson=Local_authorities,
                           locations='LAD',
                           color='Value',
                           featureidkey="properties.LAD21NM",
                           color_continuous_scale="Viridis", # Viridis
                           mapbox_style="white-bg",
                           center={"lat": 55.09621, "lon": -4.0286298},
                           zoom=4.3,
                           labels={'val':'value'})

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    # fig.write_html('plot.html', auto_open=True) 
    fig.write_image('map_'+datatype+'.pdf')

# GeoJson source: https://www.kaggle.com/datasets/dorianlazar/uk-regions-geojson/
# https://medium.com/@patohara60/interactive-mapping-in-python-with-uk-census-data-6e571c60ff4
# https://geoportal.statistics.gov.uk/
# https://www.ons.gov.uk/census/2011census/2011censusdata
# http://webarchive.nationalarchives.gov.uk/20160110193526/http:/data.statistics.gov.uk/Census/BulkdatadetailedcharacteristicsmergedwardspluslaandregE&Wandinfo3.3.zip
# dataset UK info source: https://public.opendatasoft.com/explore/dataset/georef-united-kingdom-local-authority-district/table/?flg=en-us&disjunctive.ctry_code&disjunctive.ctry_name&disjunctive.rgn_code&disjunctive.rgn_name&disjunctive.ctyua_code&disjunctive.ctyua_name&disjunctive.lad_code&disjunctive.lad_name
## datatype in ['student_num','imd_band','avg_assessment','label_score','predict_score']
# draw_map_result('result_concatenate_geo.csv','datasets/LAD_UK.csv','datasets/LAD.json','predict_score')
