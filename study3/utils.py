from scipy.io import whosmat, loadmat
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt 
import seaborn as sns 
import openai 
import re, os, sys, time, math 
from transcript_map import transcript_config_all, post_question_dict_all
from scipy.stats import pointbiserialr

class LLM_Assistent_Predict():
    def __init__(self):
        openai.api_key = 'Your OpenAI Key'
        self.messages = [ {"role": "system", "content":"You are an intelligent assistant."} ]

    def _interaction(self,input_text):
        self.messages.append({"role": "user", "content": input_text},)
        # chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.messages) # temperature=0
        chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", temperature=0, messages=self.messages) # 
        reply = chat.choices[0].message.content
        print('\n input text: \n',input_text)
        print('\n LLM answer: \n',reply)
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def _extract_scores_each(self,input_string,output_num,prefix):
        slide_scores = {}
        pattern = fr"{prefix}[-\s]*(\d+)[\s]*:[\s]*([0-9.]+)"
        matches = re.findall(pattern, input_string)
        print('matches: ',matches)
        if len(matches) != output_num:
            print('-'*80,'error because len(matches) != output_num')
            return None
        for match in matches:
            slide_id, score_str = match
            try:
                score = float(score_str)
                slide_scores[int(slide_id)] = score
            except ValueError:
                print(f"Skipping invalid score: {score_str}")
        print('slide_scores: ',slide_scores)
        return slide_scores

    def _extract_scores_predict(self,input_string):
        pattern = r'(?i)(?:Predicted|predicted)\s*:\s*(\d+\.\d+|\d+)'
        matches = re.findall(pattern, input_string)
        numeric_scores = [float(match) for match in matches]
        if len(numeric_scores) == 0: return None
        return numeric_scores[0]


    def _extract_prediction(self,predict_text,prefix,output_num = 1):
        '''
        return dictionary
        '''
        #  The output format should exactly be: Slide-ID: score., The output format should exactly be: Question-ID: 0 or 1.

        if prefix == 'Predicted:':
            return self._extract_scores_predict(predict_text)
        else:
            if output_num == 1:
                return_dict = self._extract_scores_each(predict_text,output_num,prefix)
                if return_dict == None: return None 
                return list(return_dict.values())[0]
            else:
                return self._extract_scores_each(predict_text,output_num,prefix)


    def init_task(self,task_description):
        self._interaction(task_description)

    def predict_each_student(self,persona,prefix,output_num = 1):
        prediction_answer = self._interaction(persona)
        prediction_output = self._extract_prediction(prediction_answer,prefix,output_num)
        return prediction_output


def func_LLM_predict_score(task_description,student_persona,prefix):
    LLM_Assistent_Predictor = LLM_Assistent_Predict()
    LLM_Assistent_Predictor.init_task(task_description)
    prediction = LLM_Assistent_Predictor.predict_each_student(student_persona,prefix,output_num = 1)

    while prediction == None:
        LLM_Assistent_Predictor = LLM_Assistent_Predict()
        LLM_Assistent_Predictor.init_task(task_description)
        prediction = LLM_Assistent_Predictor.predict_each_student(student_persona,prefix,output_num = 1)
    return prediction 

def func_LLM_predict_score_all(task_description,student_persona,prefix,output_num):
    LLM_Assistent_Predictor = LLM_Assistent_Predict()
    LLM_Assistent_Predictor.init_task(task_description)
    prediction = LLM_Assistent_Predictor.predict_each_student(student_persona,prefix,output_num = output_num)

    while prediction == None:
        LLM_Assistent_Predictor = LLM_Assistent_Predict()
        LLM_Assistent_Predictor.init_task(task_description)
        prediction = LLM_Assistent_Predictor.predict_each_student(student_persona,prefix,output_num = output_num)
    return prediction 

def func_generate_student_demographic(table_item):
    # exp_id,course_name,student_id,age,occupation,occupation_field,gender,study_time,degree_cat,GPA
    student_persona_demo = 'Student ' + str(int(table_item['student_id'].values[0])) + ' is a '
    student_persona_demo = student_persona_demo + str(int(table_item['age'].values[0])) + ' year old ' 
    student_persona_demo = student_persona_demo + str(table_item['gender'].values[0]) + ' student with GPA of '
    student_persona_demo = student_persona_demo + str(table_item['GPA'].values[0]) 
    if table_item['degree_cat'].values[0] == 'G': 
        student_persona_demo = student_persona_demo + ' towards graduate degree. '
    elif table_item['degree_cat'].values[0] == 'UG': 
        student_persona_demo = student_persona_demo + ' towards undergraduate degree. '
    else: 
        student_persona_demo = student_persona_demo + '. '
    student_persona_demo = student_persona_demo + 'The current occupation of this student is '
    student_persona_demo = student_persona_demo + str(table_item['occupation'].values[0]) + ' in the field of '
    student_persona_demo = student_persona_demo + str(table_item['occupation_field'].values[0]) + ' whose average study time is '
    student_persona_demo = student_persona_demo + str(table_item['study_time'].values[0]) + ' hours per day. '
    print(student_persona_demo)
    return student_persona_demo


