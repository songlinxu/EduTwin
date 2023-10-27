import numpy as np 
import pandas as pd 
import seaborn as sns
import geopandas as gpd
from matplotlib import pyplot as plt 
import openai
import os,sys,uuid,time,re,json,math 
import plotly.express as px


def concatenate_geo(dataset_path,info_path,output_path):
    student_data = pd.read_csv(dataset_path)
    info_data = pd.read_csv(info_path)
    student_data_arr = np.array(student_data)
    student_data_new = []
    for i,item in enumerate(student_data_arr):
        item_match = info_data[info_data['id_student']==item[2]]
        region = item_match['region'].values[0]
        imd_band = str(item_match['imd_band'].values[0]).split('-')[0]
        student_data_new.append([d for d in item]+[region,imd_band])

    student_data_new = pd.DataFrame(np.array(student_data_new),columns=list(student_data.columns.values)+['region','imd_band'])
    student_data_new.to_csv(output_path,index=False)

def visual_pearson_matrix(predict_result_path):

    predict_result_table = pd.read_csv(predict_result_path)

    run_id_list = ['label_score','predict_score_persona','predict_score_past','predict_score_both','predict_score_past_1','predict_score_past_2','predict_score_past_3','predict_score_past_4','predict_score_past_5']
    predict_name_dict = {'label_score': 'H','predict_score_persona': 'i','predict_score_past': 'ii','predict_score_both': 'iii','predict_score_past_1': 'iv','predict_score_past_2': 'v','predict_score_past_3': 'vi','predict_score_past_4': 'vii','predict_score_past_5': 'viii'}
    # predict_name_dict = {'label_score': 'Human','predict_score_persona': 'Type i','predict_score_past': 'Type ii','predict_score_both': 'Type iii','predict_score_past_1': 'Type iv','predict_score_past_2': 'Type v','predict_score_past_3': 'Type vi','predict_score_past_4': 'Type vii','predict_score_past_5': 'Type viii'}


    data = {}
    past_score_avg = []
    for run_id in run_id_list:
        table_item = predict_result_table[predict_result_table['score_type']==run_id]
        data[predict_name_dict[str(run_id)]] = list(table_item['score_item'].values)
        if run_id != 'label_score': continue
        table_item_arr = np.array(table_item)
        for i in range(len(table_item_arr)):
            table_item_arr_item = table_item_arr[i]
            past_score_item = table_item_arr_item[-3]
            past_score_splits = past_score_item.split('-')
            score_avg = 0
            for score in past_score_splits:
                score_avg += float(score)
            score_avg = score_avg / len(past_score_splits)

            past_score_avg.append(score_avg)

    # data['Assessment'] = past_score_avg
    data['A'] = past_score_avg

    df = pd.DataFrame(data)

    correlation_matrix = df.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.set(style="white")
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='Reds', fmt=".2f", annot_kws={"fontsize":20})
    # sns.heatmap(correlation_matrix, annot=True, cmap=sns.diverging_palette(20, 220, n=200), fmt=".2f")
    # https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
    # sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="Reds", mask=mask, square=True, cbar=True)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    plt.title('Pearson Correlation Matrix: Experiment 2', fontsize=24)
    plt.tight_layout()
    plt.savefig('corr_exp2.pdf')
    plt.show()

def reveal_corr_assess_exam(dataset_path):
    data = pd.read_csv(dataset_path)
    predict_type_list = ['label_score','predict_score_persona','predict_score_past','predict_score_both','predict_score_past_1','predict_score_past_2','predict_score_past_3','predict_score_past_4','predict_score_past_5']
    predict_name_dict = {'label_score': 'Human','predict_score_persona': 'Type i','predict_score_past': 'Type ii','predict_score_both': 'Type iii','predict_score_past_1': 'Type iv','predict_score_past_2': 'Type v','predict_score_past_3': 'Type vi','predict_score_past_4': 'Type vii','predict_score_past_5': 'Type viii'}
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for p,predict_type in enumerate(predict_type_list):
        data_item = data[data['score_type']==predict_type]
        data_arr = np.array(data_item)
        past_score_avg = []
        for i in range(len(data_arr)):
            data_arr_item = data_arr[i]
            past_score_item = data_arr_item[-3]
            past_score_splits = past_score_item.split('-')
            score_avg = 0
            for score in past_score_splits:
                score_avg += float(score)
            score_avg = score_avg / len(past_score_splits)

            past_score_avg.append(score_avg)

        past_score_avg = np.array(past_score_avg).reshape((len(past_score_avg),1))
        data_new = pd.DataFrame(np.concatenate((data_arr,past_score_avg),axis=1),columns=['course_id','course_year','student_id','past_scores','score_type','score_item','past_score_avg'])
        current_ax = axes[int(p/3)][p-int(p/3)*3]
        sns.scatterplot(data_new,x='past_score_avg',y='score_item',ax=current_ax,s=32,color='red',alpha=0.7)
        current_ax.set_xlim(0,100)
        current_ax.set_ylim(0,100)
        current_ax.set_xlabel('Past Assessments Average Score',fontsize = 14)
        current_ax.set_ylabel('Final Exam Score',fontsize = 14)
        current_ax.set_title(predict_name_dict[predict_type],fontsize = 14)
        current_ax.tick_params(axis='x', labelsize=14) 
        current_ax.tick_params(axis='y', labelsize=14) 
        current_ax.spines['top'].set_visible(False)
        current_ax.spines['right'].set_visible(False)

        # sns.scatterplot(data_new,x='past_score_avg',y='score_item',hue='score_type')
    plt.tight_layout()
    plt.savefig('reveal_corr.png')
    plt.show()

def visual_compare_main(result_file):
    result_data = pd.read_csv(result_file)

    fontsize = 16
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    course_list = ['CCC','DDD','Both']
    predict_type_list = ['predict_score_persona','predict_score_past','predict_score_both','predict_score_past_1','predict_score_past_2','predict_score_past_3','predict_score_past_4','predict_score_past_5']
    predict_name_dict = {'label_score': 'Human','predict_score_persona': 'Type i','predict_score_past': 'Type ii','predict_score_both': 'Type iii','predict_score_past_1': 'Type iv','predict_score_past_2': 'Type v','predict_score_past_3': 'Type vi','predict_score_past_4': 'Type vii','predict_score_past_5': 'Type viii'}
    for c,course_id in enumerate(course_list):
        if course_id != 'Both':
            result_data_item = result_data[(result_data['course_id']==course_id)&((result_data['score_type']=='label_score')|(result_data['score_type']=='predict_score_persona')|(result_data['score_type']=='predict_score_both'))]
        else:
            result_data_item = result_data[((result_data['score_type']=='label_score')|(result_data['score_type']=='predict_score_persona')|(result_data['score_type']=='predict_score_both'))]
        current_ax = axes[c]
        sns.lineplot(data=result_data_item,x='region',y='score_item',hue='score_type',ax=current_ax)
        current_ax.legend([])
        current_ax.set_ylim(54,86)
        current_ax.set_xlabel('Region',fontsize = fontsize)
        current_ax.set_ylabel('Final Exam Score',fontsize = fontsize)
        current_ax.set_title(course_id,fontsize = fontsize)
        # current_ax.tick_params(axis='x', labelsize=fontsize)
        current_ax.set_xticks([]) 
        current_ax.tick_params(axis='y', labelsize=fontsize) 
        current_ax.spines['top'].set_visible(False)
        current_ax.spines['right'].set_visible(False)

        # sns.scatterplot(data_new,x='past_score_avg',y='score_item',hue='score_type')
    plt.tight_layout()
    plt.savefig('visual_compare_main.pdf')

    plt.show()


def visual_compare_geo(result_file):
    result_data = pd.read_csv(result_file)

    col = 2
    fontsize = 16
    fig, axes = plt.subplots(4, 2, figsize=(9, 12))
    predict_type_list = ['predict_score_persona','predict_score_past','predict_score_both','predict_score_past_1','predict_score_past_2','predict_score_past_3','predict_score_past_4','predict_score_past_5']
    predict_name_dict = {'label_score': 'Human','predict_score_persona': 'Type i','predict_score_past': 'Type ii','predict_score_both': 'Type iii','predict_score_past_1': 'Type iv','predict_score_past_2': 'Type v','predict_score_past_3': 'Type vi','predict_score_past_4': 'Type vii','predict_score_past_5': 'Type viii'}
    for p,predict_type in enumerate(predict_type_list):
        result_data_item = result_data[(result_data['score_type']=='label_score')|(result_data['score_type']==predict_type)]
        current_ax = axes[int(p/col)][p-int(p/col)*col]
        sns.lineplot(data=result_data_item,x='region',y='score_item',hue='score_type',ax=current_ax)
        current_ax.legend([])
        current_ax.set_ylim(50,90)
        current_ax.set_xlabel('Region',fontsize = fontsize)
        current_ax.set_ylabel('Final Exam Score',fontsize = fontsize)
        current_ax.set_title(predict_name_dict[predict_type],fontsize = fontsize)
        # current_ax.tick_params(axis='x', labelsize=fontsize)
        current_ax.set_xticks([]) 
        current_ax.tick_params(axis='y', labelsize=fontsize) 
        current_ax.spines['top'].set_visible(False)
        current_ax.spines['right'].set_visible(False)

        # sns.scatterplot(data_new,x='past_score_avg',y='score_item',hue='score_type')
    plt.tight_layout()
    plt.savefig('visual_compare_geo.pdf')

    plt.show()


def _get_value_from_str(pupil_string):
    pupil_item = 0
    pupil_splits = pupil_string.split('-')

    for pupil_each in pupil_splits:
        pupil_item += float(pupil_each)
    return pupil_item/len(pupil_splits)






# concatenate_geo('result.csv','datasets/OULA/studentInfo.csv','result_concatenate_geo.csv')


# figure for main text

# visual_compare_main('result_concatenate_geo.csv')
# visual_pearson_matrix('result.csv')



# figure for appendix
# reveal_corr_assess_exam('result.csv')
# visual_compare_geo('result_concatenate_geo.csv')