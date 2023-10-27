import numpy as np 
import pandas as pd 
import seaborn as sns
import geopandas as gpd
from matplotlib import pyplot as plt 
import openai
import os,sys,uuid,time,re 

task_description = "I have a task for you. I will give you the information background and past assessment performance of each student and I hope you could predict their final exam score (numeric value from 0 to 100) according to your understanding of the importance of different backgrounds and past assessment performance related to students' success. I know these factors could not absolutely predict student final exam score but I hope you could make a prediction according to my information. You should only output one prediction numeric value from 0 to 100. You do not need to give me any reason. You should just tell me your prediction based on your understanding of different factors and past performance that may affect students' final exam performance. I will input student information in the next input."


def generate_persona(table_item):
    persona_string = 'Student'
    persona_string = persona_string + ' ' + str(table_item['id_student'].values[0]) + ' is a '
    persona_string = persona_string + 'male ' if table_item['gender'].values[0] == 'M' else persona_string + 'female '
    persona_string = persona_string + str(table_item['age_band'].values[0]) + ' year old student '
    persona_string = persona_string + 'with disability from ' if table_item['disability'].values[0] == 'Y' else persona_string + 'without disability from '
    persona_string = persona_string + str(table_item['region'].values[0]) + ' with highest education of '
    persona_string = persona_string + str(table_item['highest_education'].values[0]) + ', living in an area where the IMD band is '
    persona_string = persona_string + str(table_item['imd_band'].values[0]) + '.'
    print(persona_string)
    return persona_string


class LLM_Assistent_Predict():
    def __init__(self):
        openai.api_key = 'Your OpenAI Key'
        self.messages = [ {"role": "system", "content":"You are an intelligent assistant."} ]

    def _interaction(self,input_text):
        self.messages.append({"role": "user", "content": input_text},)
        chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", temperature=0, messages=self.messages)
        reply = chat.choices[0].message.content
        print('\n input text: \n',input_text)
        print('\n LLM answer: \n',reply)
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def _extract_prediction(self,predict_text):
        # extract numeric value from answers
        numeric_strings = re.findall(r'\d+\.\d+|\d+', predict_text)
        numeric_values = [float(s) for s in numeric_strings if 0 <= float(s) <= 100]
        if len(numeric_values) == 0: return None 
        return numeric_values[0]


    def init_task(self,task_description):
        self._interaction(task_description)

    def predict_each_student(self,persona):
        prediction_answer = self._interaction(persona)
        prediction_output = self._extract_prediction(prediction_answer)
        return prediction_output


def _get_course_info():
    course_info_file = 'datasets/OULA/courses.csv'
    course_info_data = pd.read_csv(course_info_file)
    course_list = list(set(course_info_data['code_module']))
    course_list.sort()

    assess_info_file = 'datasets/OULA/assessments.csv'
    assess_info_data = pd.read_csv(assess_info_file)

    course_info_dict = {}
    for i,course_id in enumerate(course_list):
        course_info_data_item = course_info_data[course_info_data['code_module']==course_id]
        course_year_list_item = list(set(course_info_data_item['code_presentation']))
        course_year_list_item.sort()
        course_year_dict = {}
        for j,course_year in enumerate(course_year_list_item):
            assess_year_data = assess_info_data[(assess_info_data['code_presentation']==course_year)&(assess_info_data['code_module']==course_id)]
            assess_id_list_item = list(set(assess_year_data['id_assessment']))
            assess_id_list_item.sort()
            assess_id_info_list = []
            for k,assess_id in enumerate(assess_id_list_item):
                assess_item = assess_year_data[assess_year_data['id_assessment']==assess_id]
                assess_type = assess_item['assessment_type'].values[0]
                assess_id_info_list.append((assess_id,assess_type))
            course_year_dict[course_year] = assess_id_info_list
        
        course_info_dict[course_id] = course_year_dict
    
    print(course_info_dict)
    return course_info_dict,course_list

def func_LLM_predict_score(student_persona_with_score):
    LLM_Assistent_Predictor = LLM_Assistent_Predict()
    LLM_Assistent_Predictor.init_task(task_description)
    prediction = LLM_Assistent_Predictor.predict_each_student(student_persona_with_score)

    while prediction == None:
        LLM_Assistent_Predictor = LLM_Assistent_Predict()
        LLM_Assistent_Predictor.init_task(task_description)
        prediction = LLM_Assistent_Predictor.predict_each_student(student_persona_with_score)
    return prediction 




