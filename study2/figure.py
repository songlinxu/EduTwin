import numpy as np 
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt 
import openai
import os,sys,uuid,time,re,json,math 
import plotly.express as px
from scipy.stats import pearsonr



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

def reorganize_table(predict_result_path):
    raw_data = pd.read_csv(predict_result_path)
    predict_type_list = ['label_score','predict_score_persona','predict_score_past','predict_score_both','predict_score_past_1','predict_score_past_2','predict_score_past_3','predict_score_past_4','predict_score_past_5']
    header_list = ['course_id','course_year','student_id'] + predict_type_list + ['Assessment','Region','IMD']

    data_len = len(raw_data)
    new_data = []
    
    student_id_list = list(set(raw_data['student_id']))
    region_list = list(set(raw_data['region']))
    region_list.sort()
    region_dict = {region_item:r for r,region_item in enumerate(region_list)}
    for student_id in student_id_list:
        student_data = raw_data[(raw_data['student_id']==student_id)]
        if len(student_data) == 0: continue
        assert len(student_data) % len(predict_type_list) == 0
        course_id_list = list(set(student_data['course_id']))
        course_year_list = list(set(student_data['course_year']))
        for course_id in course_id_list:
            for course_year in course_year_list:
                sub_data = student_data[(student_data['course_id']==course_id)&(student_data['course_year']==course_year)]
                if len(sub_data) == 0: continue

                predict_result_list = []
                for score_type in predict_type_list:
                    sub_data_type = sub_data[sub_data['score_type']==score_type]
                    if len(sub_data_type) == 0: break
                    predict_result_list.append(sub_data_type['score_item'].values[0])

                past_score_splits = sub_data['past_scores'].values[0].split('-')
                score_avg = 0
                for score in past_score_splits:
                    score_avg += float(score)
                score_avg = score_avg / len(past_score_splits)
                assessment = score_avg
                region = region_dict[sub_data['region'].values[0]]
                imd = sub_data['imd_band'].values[0]
                new_data.append([course_id,course_year,student_id]+predict_result_list+[assessment,region,imd])


    new_data = pd.DataFrame(np.array(new_data),columns=header_list)
    return new_data


def get_pearson_result(predict_result_path):

    predict_result_table = pd.read_csv(predict_result_path)

    run_id_list = ['label_score','predict_score_persona','predict_score_past','predict_score_both','predict_score_past_1','predict_score_past_2','predict_score_past_3','predict_score_past_4','predict_score_past_5']
    predict_name_dict = {'label_score': 'H','predict_score_persona': 'i','predict_score_past': 'ii','predict_score_both': 'iii','predict_score_past_1': 'iv','predict_score_past_2': 'v','predict_score_past_3': 'vi','predict_score_past_4': 'vii','predict_score_past_5': 'viii'}

    data = {}
    past_score_avg = []
    for run_id in run_id_list:
        table_item = predict_result_table[predict_result_table['score_type']==run_id]
        data[run_id] = list(table_item['score_item'].values)
        if run_id != 'label_score': continue
        table_item_arr = np.array(table_item)
        for i in range(len(table_item_arr)):
            table_item_arr_item = table_item_arr[i]
            past_score_item = table_item_arr_item[-5]
            past_score_splits = past_score_item.split('-')
            score_avg = 0
            for score in past_score_splits:
                score_avg += float(score)
            score_avg = score_avg / len(past_score_splits)

            past_score_avg.append(score_avg)

    data['Assessment'] = past_score_avg
    # data['A'] = past_score_avg

    df = pd.DataFrame(data)

    return df







def _get_value_from_str(pupil_string):
    pupil_item = 0
    pupil_splits = pupil_string.split('-')
    len_item = 0

    for pupil_each in pupil_splits:
        if pupil_each == 'nan': continue
        pupil_item += float(pupil_each)
        len_item += 1
    return pupil_item/len_item

def func_discretize_assessment(raw_value,delta=85.625):
    if raw_value < 0.5*delta:
        return 'Low'
    else:
        return 'High'

def check_trend_with_correlation(pupil_string):
    pupil_splits = pupil_string.split('-')
    pupil_list = []

    for pupil_each in pupil_splits:
        if pupil_each == 'nan': continue
        pupil_list.append(float(pupil_each))

    arr = np.array(pupil_list)

    index = np.arange(len(arr))
    correlation_coefficient, _ = pearsonr(index, arr)

    if correlation_coefficient > 0:
        return 'Increase  \u2191'
    else:
        return 'Decrease \u2193'

def calculate_corr(data,target_factor):
    data_new = {}
    predict_type_list = list(set(data['score_type']))
    choice_list = list(set(data[target_factor]))
    choice_list.sort()
    for predict_type in predict_type_list:
        data_new[str(predict_type)] = []
        for c,choice in enumerate(choice_list):
            table_item = data[(data['score_type']==predict_type)&(data[target_factor]==choice)]
            data_new[str(predict_type)].append(np.mean(table_item['score_item'].values))
                

    df = pd.DataFrame(data_new)
    print(target_factor,df)

    pearson_r_dict = {}
    for col1 in df.columns:
        for col2 in df.columns:
            # if col1 == 'label_score':
            if col1 == 'label_score' and col2 != 'label_score':
                # print(df[col1],df[col2])
                pearson_r_temp, _ = pearsonr(df[col1], df[col2])
                pearson_r_dict[col2] = round(pearson_r_temp,2)

    return pearson_r_dict

def compare_distribution_assessment(result_file):
    dataset = pd.read_csv(result_file) 
    dataset = dataset.dropna()
    dataset['past_score_raw'] = dataset['past_scores'].apply(_get_value_from_str)
    past_score_arr = np.array(dataset['past_score_raw'])
    p_max, p_min = np.max(past_score_arr), np.min(past_score_arr)
    print(p_max, p_min)
    dataset['past_score_value'] = dataset['past_score_raw'].apply(func_discretize_assessment)
    sns.barplot(data=dataset,x='score_type',y='score_item',hue='past_score_value')
    plt.show()


def _change_score_name(raw_name):
    name_dict = {'predict_score_both': 'Both', 'predict_score_past': 'Past', 'predict_score_persona': 'persona','label_score': 'Human','predict_score_past_1': '1','predict_score_past_2': '2', 'predict_score_past_3': '3', 'predict_score_past_4': '4', 'predict_score_past_5': '5'}
    return name_dict[raw_name]

def f2_main(result_file):
    pearson_table = reorganize_table(result_file)
    result_data = pd.read_csv(result_file)
    fontsize = 8
    fig, axes = plt.subplots(2, 3, figsize=(16, 7))


    for iii in range(2):
        for jjj in range(3):
            if iii == 1 and jjj == 0: continue
            axes[iii][jjj].spines['top'].set_visible(False)
            axes[iii][jjj].spines['right'].set_visible(False)

    label_dict = {'label_score':'$Label$','predict_score_persona':'$T_{demo}$','predict_score_past':'$T_{past}$','predict_score_both':'$T_{both}$','predict_score_past_1':'$T_{p1}$','predict_score_past_5':'$T_{p5}$','predict_score_past_2':'$T_{p2}$','predict_score_past_3':'$T_{p3}$','predict_score_past_4':'$T_{p4}$'}

    pearson_table_sub_1 = pearson_table[['label_score','predict_score_persona','predict_score_past','predict_score_both']]
    correlation_matrix = pearson_table_sub_1.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool),k=1)   
    correlation_matrix_renamed = correlation_matrix.rename(index=label_dict, columns=label_dict)
    heatmap = sns.heatmap(correlation_matrix_renamed, annot=True, cmap='Oranges', fmt=".2f", mask=mask, annot_kws={"fontsize":12},ax=axes[1][0])
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    axes[1][0].tick_params(axis='both', which='major', labelsize=13) 

    predict_type_name_dict = label_dict

    result_data['Model'] = result_data['score_type'].replace(label_dict)
    sns.ecdfplot(result_data,x='score_item',hue='Model',ax=axes[1][2])
    axes[1][2].set_xlabel('Final Exam Score (Prediction/Label)')
    axes[1][2].set_ylabel('Cumulative Probability')

    pearson_region_dict = calculate_corr(result_data,'region')

    predict_type_list = ['label_score','predict_score_persona','predict_score_past','predict_score_both','predict_score_past_1','predict_score_past_2','predict_score_past_3','predict_score_past_4','predict_score_past_5']
    float_type_list = ['IMD','Region','Assessment'] + predict_type_list

    for float_type in float_type_list:
        pearson_table[float_type] = pearson_table[float_type].astype(float)
        
    pearson_r_0, _ = pearsonr(pearson_table['label_score'], pearson_table['predict_score_persona'])
    pearson_r_1, _ = pearsonr(pearson_table['label_score'], pearson_table['predict_score_past_1'])
    pearson_r_2, _ = pearsonr(pearson_table['label_score'], pearson_table['predict_score_past_2'])
    pearson_r_3, _ = pearsonr(pearson_table['label_score'], pearson_table['predict_score_past_3'])
    pearson_r_4, _ = pearsonr(pearson_table['label_score'], pearson_table['predict_score_past_4'])
    pearson_r_5, _ = pearsonr(pearson_table['label_score'], pearson_table['predict_score_past_5'])
    pearson_r_6, _ = pearsonr(pearson_table['label_score'], pearson_table['predict_score_past'])
    pearson_r_7, _ = pearsonr(pearson_table['label_score'], pearson_table['predict_score_both'])

    pearson_table = pearson_table.dropna()
    pearson_u_label, _ = pearsonr(pearson_table['Assessment'], pearson_table['label_score'])
    pearson_u_0, _ = pearsonr(pearson_table['Assessment'], pearson_table['predict_score_persona'])
    pearson_u_1, _ = pearsonr(pearson_table['Assessment'], pearson_table['predict_score_past_1'])
    pearson_u_2, _ = pearsonr(pearson_table['Assessment'], pearson_table['predict_score_past_2'])
    pearson_u_3, _ = pearsonr(pearson_table['Assessment'], pearson_table['predict_score_past_3'])
    pearson_u_4, _ = pearsonr(pearson_table['Assessment'], pearson_table['predict_score_past_4'])
    pearson_u_5, _ = pearsonr(pearson_table['Assessment'], pearson_table['predict_score_past_5'])
    pearson_u_6, _ = pearsonr(pearson_table['Assessment'], pearson_table['predict_score_past'])
    pearson_u_7, _ = pearsonr(pearson_table['Assessment'], pearson_table['predict_score_both'])

   
    result_data_filter = result_data.dropna()
    print(set(result_data_filter['score_type']))
    specific_score_types = ['predict_score_persona','predict_score_past_1','predict_score_past_2','predict_score_past_3','predict_score_past_4','predict_score_past_5','predict_score_both']
    line_styles = [':' if score_type in specific_score_types else '-' for score_type in result_data_filter['score_type'].unique()]
    sns.pointplot(result_data_filter,x='imd_band',y='score_item',hue='score_type',linewidth=1,scale=0.65,markers='o', linestyles=line_styles,ax=axes[0][2])
    pearson_imd_dict = calculate_corr(result_data_filter,'imd_band')
    handles, labels = axes[0][2].get_legend_handles_labels()
    new_labels = [predict_type_name_dict[label] if label in predict_type_name_dict else label for label in labels]
    axes[0][2].set_xlabel('IMD Band')
    axes[0][2].set_ylabel('Final Exam Score (Prediction/Label)')
    axes[0][2].set_xticklabels([str(int(ti*10)) for ti in range(10)])
    axes[0][2].legend(handles, new_labels, bbox_to_anchor=(0.5, 1.2), title='Model', loc='upper center', ncol=5, fontsize=8)

    for artist in axes[0][2].collections:
        artist.set_edgecolor(artist.get_facecolor())  # Set edge color to match face color
        artist.set_facecolor('none')  # Set face color to none to make markers hollow

    
    sns.pointplot(x=[1,2,3,4,5],y=[pearson_r_1,pearson_r_2,pearson_r_3,pearson_r_4,pearson_r_5],color='#2ca02c',alpha=1,ax=axes[1][1],label='$r_{overall}$') 
    sns.pointplot(x=[6,7],y=[pearson_r_6,pearson_r_7],color='#2ca02c',alpha=1,ax=axes[1][1]) 

    sns.pointplot(x=[1,2,3,4,5],y=[pearson_u_1,pearson_u_2,pearson_u_3,pearson_u_4,pearson_u_5],color='#1f77b4',alpha=1,ax=axes[1][1],label='$r_{assessment}$')
    sns.pointplot(x=[6,7],y=[pearson_u_6,pearson_u_7],color='#1f77b4',alpha=1,ax=axes[1][1]) 
    sns.pointplot(x=[1,2,3,4,5],y=[pearson_region_dict['predict_score_past_1'],pearson_region_dict['predict_score_past_2'],pearson_region_dict['predict_score_past_3'],pearson_region_dict['predict_score_past_4'],pearson_region_dict['predict_score_past_5']],color='#ff7f0e',alpha=1,ax=axes[1][1])
    sns.pointplot(x=[6,7],y=[pearson_region_dict['predict_score_past'],pearson_region_dict['predict_score_both']],color='#ff7f0e',alpha=1,ax=axes[1][1],label='$r_{per-region}$') 
    sns.pointplot(x=[1,2,3,4,5],y=[pearson_imd_dict['predict_score_past_1'],pearson_imd_dict['predict_score_past_2'],pearson_imd_dict['predict_score_past_3'],pearson_imd_dict['predict_score_past_4'],pearson_imd_dict['predict_score_past_5']],color='#808080',alpha=1,ax=axes[1][1],label='$r_{per-IMD-Band}$')
    sns.pointplot(x=[6,7],y=[pearson_imd_dict['predict_score_past'],pearson_imd_dict['predict_score_both']],color='#808080',alpha=1,ax=axes[1][1]) 
    axes[1][1].set_xlabel('Model')
    axes[1][1].set_ylabel('Pearson $r$')
    dot_label_dict = {1:'$T_{p1}$',2:'$T_{p2}$',3:'$T_{p3}$',4:'$T_{p4}$',5:'$T_{p5}$',6:'$T_{past}$',7:'$T_{both}$',}
    axes[1][1].set_xticklabels([dot_label_dict[di+1] for di in range(7)])
    handles, labels = axes[1][1].get_legend_handles_labels()
    axes[1][1].legend(handles, labels, bbox_to_anchor=(0, 1), frameon=False, title='', loc='upper left', ncol=2, fontsize=8)

    result_data_copy = pd.DataFrame(np.array(result_data),columns=result_data.columns.values)
    result_data_copy['past_score_raw'] = result_data_copy['past_scores'].apply(_get_value_from_str) 
    result_data_copy['Past Assessment Score Trend'] = result_data_copy['past_scores'].apply(check_trend_with_correlation) 
    result_data_copy['score_name'] = result_data_copy['score_type'].replace(label_dict)
    past_score_arr = np.array(result_data_copy['past_score_raw'])
    p_max, p_min = np.max(past_score_arr), np.min(past_score_arr)
    result_data_copy['Past Assessment Mean Score'] = result_data_copy['past_score_raw'].apply(func_discretize_assessment, args=(p_max - p_min,))

    result_data_copy['score_item'] = result_data_copy['score_item'].astype(float)

    sns.stripplot(data=result_data_copy, x="score_name", y="score_item", hue="Past Assessment Mean Score",size=1.3,dodge=True, alpha=.5, legend=False,ax=axes[0][1])
    sns.pointplot(data=result_data_copy, x="score_name", y="score_item", hue="Past Assessment Mean Score",dodge=0.4, linestyle="none", errorbar=None,marker="_", markersize=15, markeredgewidth=3,ax=axes[0][1])
    axes[0][1].set_xlabel('Model')
    axes[0][1].set_ylabel('Final Exam Score (Prediction/Label)')
    axes[0][1].legend(bbox_to_anchor=(0.5, 1.2), title='Past Assessment Average Score', loc='upper center', ncol=2, fontsize=10)

    sns.stripplot(data=result_data_copy, x="score_name", y="score_item", hue="Past Assessment Score Trend",size=1.3,dodge=True, alpha=.5, legend=False,ax=axes[0][0])
    sns.pointplot(data=result_data_copy, x="score_name", y="score_item", hue="Past Assessment Score Trend",dodge=0.4, linestyle="none", errorbar=None,marker="_", markersize=15, markeredgewidth=3,ax=axes[0][0])
    axes[0][0].set_xlabel('Model')
    axes[0][0].set_ylabel('Final Exam Score (Prediction/Label)')
    axes[0][0].legend(bbox_to_anchor=(0.5, 1.2), title='Past Assessment Score Trend', loc='upper center', ncol=2, fontsize=10)

    plt.subplots_adjust(wspace=0.2)
    plt.savefig('f2_main.pdf')

    plt.show()



# concatenate_geo('result.csv','datasets/OULA/studentInfo.csv','result_concatenate_geo.csv')


# figure for main text

f2_main('result_concatenate_geo.csv')
