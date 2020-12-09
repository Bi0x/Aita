#! /usr/bin/python3
from sklearn.cluster import KMeans
import jieba
import re
import collections

def read_remove_word():
    # ? 获取需要剔除的字符和词语
    fp = open('./assets/remove_words.txt', 'r')
    return ''.join(fp.readlines()).split('|')
    

def show_question_name():
    # ? 显示问题
    fp = open('./assets/question.txt')
    print("针对的问题：《" + fp.readline()[:-1] + "》")
    fp.close()


def train_file_reader():
    # ? 读取答案数据，并将答案分组
    train_file = open('./assets/marked_answers.txt')
    file_text = ""
    for line in train_file.readlines():
        file_text += line
    regex_pattern = r'(\d{12}.*?#Total: *?\d\.*\d*/10)'
    reg = re.compile(regex_pattern, re.DOTALL)
    return reg.findall(file_text)


def get_total_score(answers):
    # ? 分析答案，获取评分
    total_score = []
    regex_pattern = r'#Total: *?(\d\.*\d*)/10'
    reg = re.compile(regex_pattern, re.DOTALL)
    for answer in answers:
        total_score.append(float(reg.findall(answer)[0]))
    return total_score


def get_answers_main(answers):
    # ? 获取主要回答内容
    answers_main = []
    regex_pattern = r':\d\d\n(.*)?#Marking Scheme'
    reg = re.compile(regex_pattern, re.DOTALL)
    for answer in answers:
        answers_main.append(reg.findall(answer)[0])
    return answers_main


def get_answers_len(answers):
    # ? 获取回答长度
    answers_len = []
    for answer in answers:
        answers_len.append(len(answer))
    return answers_len


def keyword_analyse(answers):
    # ? 关键词分析
    remove_words = read_remove_word()
    word_objects = []
    for answer in answers:
        word_list = jieba.cut(answer, cut_all=False)
        for i in word_list:
            if i not in remove_words:
                word_objects.append(i)
    word_counts = collections.Counter(word_objects)
    word_counts_top10 = word_counts.most_common(50)
    print(word_counts_top10)


def train():
    # ? 训练因子选择：[字数, 关键词统计]
    marked_answers = train_file_reader()
    total_score = get_total_score(marked_answers)
    main_answers = get_answers_main(marked_answers)
    answers_len = get_answers_len(marked_answers)
    keyword_analyse(main_answers)


if __name__ == '__main__':
    # show_question_name()
    train()
    