import numpy as np 
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt 
import openai
import os,sys,uuid,time,math 
import plotly.express as px
from scipy.stats import pearsonr
from mpl_toolkits.axes_grid1 import make_axes_locatable


from utils import dataset_config_label, dataset_config_label_draw

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


def visual_factor_all(dataset_path):
    # age,gender,highschool,scholarship,work,activity,partner,salary,transport,living,mother_edu,father_edu,sibling_num,parental_status,mother_job,father_job,study_hour,read_freq_no_sci,read_freq_sci,attend_dept,impact_project,attend_class,prep_study,prep_exam,note,listen,discuss,classroom
    factors_list = ['age','gender','highschool','scholarship','work','activity','partner','salary','transport','living','mother_edu','father_edu','sibling_num','parental_status','mother_job','father_job','study_hour','read_freq_no_sci','read_freq_sci','attend_dept','impact_project','attend_class','prep_study','prep_exam','note','listen','discuss','classroom']
    # factors_list = ['age','transport','living','mother_job','read_freq_no_sci','note','listen','discuss']
    # factors_list = ['age','activity','salary','transport','living','mother_edu','father_edu','sibling_num','parental_status','mother_job','father_job','study_hour','read_freq_no_sci','read_freq_sci','attend_class','prep_exam','note','listen','discuss','classroom']
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
    data_convert = pd.DataFrame(np.array(data_convert),columns=factors_list+['run_id','predict','student_id'])
    data_convert['predict'] = data_convert['predict'].astype(float)
    sns.set(style="white")

    # row_n = int(math.sqrt(len(factors_list)))+1
    # fig, axes = plt.subplots(row_n-1, row_n, figsize=(16, 8))
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

        sns.lineplot(x=factor,y="predict",hue="run_id",palette=custom_palette,data=data,marker="o",markersize=12,alpha=0.7,errorbar=None,ax=axes[int(i/col_n)][i-int(i/col_n)*col_n],legend=False)
        # sns.lineplot(x=factor,y="predict",data=data,marker="o",errorbar=None,ax=axes[int(i/row_n)][i-int(i/row_n)*row_n])
        # axes[int(i/row_n)][i-int(i/row_n)*row_n].set_title(factor)

        # custom_xticks = [1, 2, 3, 4, 5]  # Replace with your desired ticks
        # custom_xlabels = ['Label1', 'Label2', 'Label3', 'Label4', 'Label5']  # Replace with your desired labels
        # axes[int(i/row_n)][i-int(i/row_n)*row_n].set_xticks(custom_xticks)
        # axes[int(i/row_n)][i-int(i/row_n)*row_n].set_xticklabels(custom_xlabels)
        axes[int(i/col_n)][i-int(i/col_n)*col_n].set_xlabel(factor, fontsize=20)
        axes[int(i/col_n)][i-int(i/col_n)*col_n].set_ylabel('Grade', fontsize=20)
        axes[int(i/col_n)][i-int(i/col_n)*col_n].tick_params(axis='x', labelsize=20) 
        axes[int(i/col_n)][i-int(i/col_n)*col_n].tick_params(axis='y', labelsize=20) 
        axes[int(i/col_n)][i-int(i/col_n)*col_n].spines['top'].set_visible(False)
        axes[int(i/col_n)][i-int(i/col_n)*col_n].spines['right'].set_visible(False)
        axes[int(i/col_n)][i-int(i/col_n)*col_n].spines['bottom'].set_visible(False)
        axes[int(i/col_n)][i-int(i/col_n)*col_n].set_title('r = '+str(pearson_r), fontsize=20)

    

    plt.tight_layout()
    # plt.subplots_adjust(hspace=0.2, wspace=0.1)
    plt.savefig('factor_line.pdf')
    plt.show()


def visual_correlation_all(dataset_path):
    # age,gender,highschool,scholarship,work,activity,partner,salary,transport,living,mother_edu,father_edu,sibling_num,parental_status,mother_job,father_job,study_hour,read_freq_no_sci,read_freq_sci,attend_dept,impact_project,attend_class,prep_study,prep_exam,note,listen,discuss,classroom
    factors_list = ['age','gender','highschool','scholarship','work','activity','partner','salary','transport','living','mother_edu','father_edu','sibling_num','parental_status','mother_job','father_job','study_hour','read_freq_no_sci','read_freq_sci','attend_dept','impact_project','attend_class','prep_study','prep_exam','note','listen','discuss','classroom']
    # factors_list = ['age','activity','salary','transport','living','mother_edu','father_edu','sibling_num','parental_status','mother_job','father_job','study_hour','read_freq_no_sci','read_freq_sci','attend_class','prep_exam','note','listen','discuss','classroom']
    data = pd.read_csv(dataset_path)
    data_arr = np.array(data)
    
    sns.set(style="white")

    run_id_list = list(set(data['run_id']))
    run_id_list.sort()

    col_n = 4
    fig, axes = plt.subplots(7, 4, figsize=(14,18))
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

        correlation_matrix = df.corr()
        print(factor,data_new,correlation_matrix)
        heatmap = sns.heatmap(correlation_matrix, annot=True, annot_kws={"fontsize":15}, cmap='Reds', fmt=".2f", ax=axes[int(i/col_n)][i-int(i/col_n)*col_n], cbar=False)
        axes[int(i/col_n)][i-int(i/col_n)*col_n].set_title(factor,fontsize=24)
        axes[int(i/col_n)][i-int(i/col_n)*col_n].tick_params(axis='x', labelsize=20) 
        axes[int(i/col_n)][i-int(i/col_n)*col_n].tick_params(axis='y', labelsize=20) 

    # cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # Adjust the position and size of the color bar
    # cbar = plt.colorbar(heatmap1.get_children()[0], cax=cbar_ax)  # Use the first heatmap for the color bar
    # cbar.set_label('Color Bar Label', fontsize=12)  # You can adjust the label and fontsize

    plt.tight_layout()
    # plt.subplots_adjust(hspace=0.2, wspace=0.1)
    plt.savefig('corr.pdf')
    plt.show()



def visual_demo(dataset_path):
    dataset = pd.read_csv(dataset_path)
    factor_list = list(dataset.columns.values)
    factor_list = factor_list[1:]

    num_cols_per_row = 4

    # Calculate the number of rows needed
    num_plots = len(factor_list)
    num_rows = (num_plots + num_cols_per_row - 1) // num_cols_per_row

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols_per_row, figsize=(15, 2.5 * num_rows))
    axes = axes.ravel()

    # Iterate through the columns and create distribution plots
    for i, column in enumerate(factor_list):
        ax = axes[i]
        sns.kdeplot(dataset[column], ax=ax, fill=True, color='green', lw=0)
        # ax.set_title(column, fontsize=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(column, fontsize=20)
        ax.set_ylabel('Density', fontsize=20)
        ax.tick_params(axis='x', labelsize=20) 
        ax.tick_params(axis='y', labelsize=20) 


    # Remove any empty subplots
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig('demo.pdf')
    plt.show()

# visual_factor_all('result_concatenate.csv')
# visual_correlation_all('result_concatenate.csv')
# draw_sunburst('datasets/student_prediction.csv')
# visual_demo('datasets/student_prediction.csv')
