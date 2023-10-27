import numpy as np 
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt 
import openai
import os,sys,uuid,time,math 

from utils import generate_persona, LLM_Assistent_Predict, concatenate_result

def experiment_1_run(task_description,dataset_path,output_file,repeat_num = 5):
    if os.path.exists(output_file) == False:
        with open(output_file, "a+") as file1:
            file1.write('run_id,sample_id,predict,label\n')
    existing_result = pd.read_csv(output_file)
    with open(output_file, 'r') as filer:
        existing_result_string = filer.readlines()
    if len(existing_result_string) == 0:
        with open(output_file, "a+") as file1:
            file1.write('run_id,sample_id,predict,label\n')
    data = pd.read_csv(dataset_path)
    student_list = list(set(data['student_id']))
    student_list.sort()
    for i in range(-1,repeat_num):
        for h,student_id in enumerate(student_list):
            previous_result = existing_result[(existing_result['run_id']==i)&(existing_result['sample_id']==student_id)]
            if len(previous_result) != 0: continue
            data_sample = data[data['student_id']==student_id]
            label = data_sample['grade'].values[0]
            if i == -1: 
                prediction = label
            else:
                student_persona = generate_persona(data_sample)
                LLM_Assistent_Predictor = LLM_Assistent_Predict()
                LLM_Assistent_Predictor.init_task(task_description)
                prediction = LLM_Assistent_Predictor.predict_each_student(student_persona)
                while prediction == 'None':
                    LLM_Assistent_Predictor = LLM_Assistent_Predict()
                    LLM_Assistent_Predictor.init_task(task_description)
                    prediction = LLM_Assistent_Predictor.predict_each_student(student_persona)
            with open(output_file, "a+") as file1:
                file1.write(str(i)+','+student_id+','+str(prediction)+','+str(label)+'\n')


# Note we do not use students' past GPA for input. Because we do not have detailed information related to GPA. Therefore, GPA data has no specific meanings and may lead to result bias.

task_description = "I have a task for you. I will give you the information background of each student and I hope you could predict their final grade (one from 0: Fail, 1: DD, 2: DC, 3: CC, 4: CB, 5: BB, 6: BA, 7: AA) according to your understanding of the importance of different backgrounds related to students' success. You should only output one prediction out of 0: Fail, 1: DD, 2: DC, 3: CC, 4: CB, 5: BB, 6: BA, 7: AA. You do not need to give me any reason. You should just tell me your prediction based on your understanding of different factors that may affect students' learning performance."
dataset_path = 'datasets/student_prediction.csv'

# experiment_1_run(task_description,dataset_path,'result.csv',repeat_num = 4)
# concatenate_result(dataset_path,'result.csv','result_concatenate.csv')

