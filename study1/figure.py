import numpy as np 
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt 
import openai
import os,sys,uuid,time,math 
import plotly.express as px
from scipy.stats import pearsonr
from mpl_toolkits.axes_grid1 import make_axes_locatable


from utils import dataset_config_label, dataset_config_label_draw, dataset_factor_label

def draw_sunburst(dataset_path):
    # student_id,age,gender,highschool,scholarship,work,activity,partner,salary,transport,living,mother_edu,father_edu,sibling_num,parental_status,mother_job,father_job,study_hour,read_freq_no_sci,read_freq_sci,attend_dept,impact_project,attend_class,prep_study,prep_exam,note,listen,discuss,classroom,cuml_gpa,exp_gpa,course_id,grade
    # https://plotly.com/python/sunburst-charts/#:~:text=Sunburst%20plots%20visualize%20hierarchical%20data,added%20to%20the%20outer%20rings.
    df = pd.read_csv(dataset_path)
    header_list = list(df.columns.values)
    student_list = list(set(df['student_id']))
    student_list.sort()
    df_text = []
    for i,student_id in enumerate(student_list):
        df_item = df[df['student_id']==student_id]
        df_sub = []
        for f,factor in enumerate(header_list):
            if factor == 'student_id':
                df_sub.append(student_id)
            else:
                df_sub.append(dataset_config_label_draw[factor][df_item[factor].values[0]])
        df_text.append(df_sub)
    df_text = pd.DataFrame(np.array(df_text),columns=header_list)
    path_list = header_list[1:6]
    fig = px.sunburst(df_text, path=path_list, values='grade',
                  color='grade', hover_data=['grade'],
                  color_continuous_scale='RdBu',
                  color_continuous_midpoint=np.average(df['grade'], weights=df['grade']))


    fig.write_html('demo_sunburst.html', auto_open=True) 
    fig.write_image('demo_sunburst.pdf',width=1000, height=800)


def visual_factor_all(dataset_path,fontsize=16):
    factors_list = ['age','gender','highschool','scholarship','work','activity','partner','salary','transport','living','mother_edu','father_edu','sibling_num','parental_status','mother_job','father_job','study_hour','read_freq_no_sci','read_freq_sci','attend_dept','impact_project','attend_class','prep_study','prep_exam','note','listen','discuss','classroom']
    data = pd.read_csv(dataset_path)
    data_arr = np.array(data)
    data_convert = []
    run_id_list = list(set(data['run_id']))
    run_id_list.sort()
    for d in range(len(data)):
        data_item = pd.DataFrame(data_arr[d].reshape((1,data.shape[1])),columns=data.columns.values)
        target_list = []
        for f in (factors_list+['run_id','predict','student_id']):
            if f in ['run_id','predict','student_id']:
                target = data_item[f].values[0]
            else:
                target = dataset_config_label[f][data_item[f].values[0]]
            target_list.append(target)

        data_convert.append(target_list)
    data_convert = pd.DataFrame(np.array(data_convert),columns=[dataset_factor_label[fac] for fac in factors_list]+['run_id','predict','student_id'])
    data_convert['predict'] = data_convert['predict'].astype(float)
    data_convert['run_id'] = data_convert['run_id'].astype(int)
    sns.set(style="white")

    col_n = 4
    row_n = int(len(factors_list)/col_n)
    fig, axes = plt.subplots(row_n, 4, figsize=(15, 2.5*row_n))
    custom_palette = {-1: '#f2b05a', 0: '#3074c7', 1: '#3074c7', 2: '#3074c7', 3: '#3074c7'}
    for i,factor in enumerate(factors_list):
        data_new = {}
        choice_list = list(set(data[factor]))
        choice_list.sort()
        for run_id in run_id_list:
            data_new[str(run_id)] = []
            for c,choice in enumerate(choice_list):
                table_item = data[(data['run_id']==run_id)&(data[factor]==choice)]
                data_new[str(run_id)].append(np.mean(table_item['predict'].values))

        df = pd.DataFrame(data_new)

        pearson_r_list = []
        for col1 in df.columns:
            for col2 in df.columns:
                if col1 == '-1' and col2 != '-1':
                    # print(df[col1],df[col2])
                    pearson_r_temp, _ = pearsonr(df[col1], df[col2])
                    pearson_r_list.append(pearson_r_temp)
        pearson_r = round(np.mean(pearson_r_list),2)

        sns.lineplot(x=dataset_factor_label[factor],y="predict",hue="run_id",palette=custom_palette,data=data_convert,marker="o",markersize=12,alpha=0.7,errorbar=None,ax=axes[int(i/col_n)][i-int(i/col_n)*col_n],legend=False)
        axes[int(i/col_n)][i-int(i/col_n)*col_n].set_xlabel(dataset_factor_label[factor], fontsize=fontsize)
        axes[int(i/col_n)][i-int(i/col_n)*col_n].set_ylabel('Grade', fontsize=fontsize)
        axes[int(i/col_n)][i-int(i/col_n)*col_n].tick_params(axis='x', labelsize=fontsize) 
        axes[int(i/col_n)][i-int(i/col_n)*col_n].tick_params(axis='y', labelsize=fontsize) 
        axes[int(i/col_n)][i-int(i/col_n)*col_n].spines['top'].set_visible(False)
        axes[int(i/col_n)][i-int(i/col_n)*col_n].spines['right'].set_visible(False)
        axes[int(i/col_n)][i-int(i/col_n)*col_n].spines['bottom'].set_visible(False)
        axes[int(i/col_n)][i-int(i/col_n)*col_n].set_title('$r$ = '+str(pearson_r), fontsize=fontsize)

    letter_ids = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

    for i, letter in enumerate(letter_ids):
        axes[i, 0].text(-0.5, 1.1, letter, fontsize=30, transform=axes[i, 0].transAxes)


    # plt.tight_layout()
    plt.subplots_adjust(hspace=1, wspace=0.3)
    plt.savefig('factor_line.pdf')
    # plt.show()



# visual_factor_all('result_concatenate.csv')
# draw_sunburst('datasets/student_prediction.csv')


