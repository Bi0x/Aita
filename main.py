#! /usr/bin/python3
from sklearn.cluster import KMeans
import re

from WordAnalyser import WordAnalyser

# ! Trainer Definition
class Trainer:

    # ? answers_chunk   ->      每个人的回答与评分整体内容
    # ? answers_score    ->      每个人的分数
    # ? answers_main     ->      每个人的回答的文字内容
    # ? answers_len      ->      每个人的回答的文字内容的长度
    # ? 训练因子选择：[字数, 关键词统计]
    def __init__(self):
        super().__init__()
        self.answers_chunk = self.train_file_reader()
        self.answers_score = self.get_total_score(self.answers_chunk)
        self.answers_main = self.get_answers_main(self.answers_chunk)
        self.answers_len = self.get_answers_len(self.answers_chunk)

    def show_question_name(self):
        # ? 显示问题
        fp = open('./assets/question.txt')
        print("针对的问题：《" + fp.readline()[:-1] + "》")
        fp.close()

    def train_file_reader(self):
        # ? 读取答案数据，并将答案分组
        train_file = open('./assets/marked_answers.txt')
        file_text = ""
        for line in train_file.readlines():
            file_text += line
        regex_pattern = r'(\d{12}.*?#Total: *?\d\.*\d*/10)'
        reg = re.compile(regex_pattern, re.DOTALL)
        return reg.findall(file_text)

    def get_total_score(self, answers):
        # ? 分析答案，获取评分
        total_score = []
        regex_pattern = r'#Total: *?(\d\.*\d*)/10'
        reg = re.compile(regex_pattern, re.DOTALL)
        for answer in answers:
            total_score.append(float(reg.findall(answer)[0]))
        return total_score

    def get_answers_main(self, answers):
        # ? 获取主要回答内容
        answers_main = []
        regex_pattern = r':\d\d\n(.*)?#Marking Scheme'
        reg = re.compile(regex_pattern, re.DOTALL)
        for answer in answers:
            answers_main.append(reg.findall(answer)[0])
        return answers_main

    def get_answers_len(self, answers):
        # ? 获取回答长度
        answers_len = []
        for answer in answers:
            answers_len.append(len(answer))
        return answers_len


# ! Trainer Definition End

def train(self):
    trainer = Trainer()

    trainer.keyword_analyse(trainer.answers_main)

if __name__ == '__main__':
    # show_question_name()
    train()
    