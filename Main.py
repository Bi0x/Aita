#! /usr/bin/python3
from ast import Num
from os import fpathconf
from WordAnalyser import WordAnalyser
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import re

# ! Trainer Definition
class Aita:

    # ? answers_chunk           ->      每个人的回答与评分整体内容
    # ? answers_score           ->      每个人的分数
    # ? answers_main            ->      每个人的回答的文字内容
    # ? answers_len             ->      每个人的回答的文字内容的长度
    # ? answers_keywordcount    ->      每个人的回答的文字内容中相关关键字出现次数
    # ? classifier              ->      训练得到的分类器
    # ? 训练因子选择：[字数, 高频关键词统计...]

    def __init__(self):
        super().__init__()
        word_analyser = WordAnalyser()
        self.show_question_name()
        self.answers_chunk          =   self.file_reader('./assets/marked_answers.txt')
        self.answers_score          =   self.get_total_score(self.answers_chunk)
        self.answers_main           =   self.get_answers_main(self.answers_chunk)
        self.answers_len            =   self.get_answers_len(self.answers_chunk)
        self.top_words              =   word_analyser.keyword_analyse(self.answers_main)
        self.answers_keywordcount   =   self.get_keyword_counts(self.top_words, self.answers_main)
        self.classifier             =   None

    def show_question_name(self):
        # ? 显示问题
        fp = open('./assets/question.txt')
        print("针对的问题：《" + fp.readline()[:-1] + "》")
        fp.close()

    def file_reader(self, path):
        # ? 读取答案数据，并将答案分组
        train_file = open(path)
        file_text = "".join(train_file.readlines())
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

    def get_keyword_counts(self, keywords, answers):
        keyword_counts = []
        for answer in answers:
            per_keyword_counts = []
            for i in keywords:
                per_keyword_counts.append(answer.count(i[0]))
            keyword_counts.append(per_keyword_counts)
        return keyword_counts

    def main_trainer(self):
        train_data = []
        for i in range(len(self.answers_main)):
            per_line = [self.answers_len[i]]
            per_line.extend(self.answers_keywordcount[i])
            train_data.append(per_line)
        print("------>>> 正在训练模型 <<<------")
        clf = RandomForestRegressor(max_depth=10, n_estimators=1000, min_samples_split=2)
        #print(len(train_data[0]))
        #print(self.answers_score)
        clf.fit(train_data, self.answers_score)
        self.classifier = clf
        print("------>>> 训练模型结束 <<<------")
    
    def predict(self, predict_answers):
        print("------>>> 预测结果如下 <<<------")
        predict_keywords_count = self.get_keyword_counts(self.top_words, [predict_answers])
        predict_data = self.get_answers_len([predict_answers])
        predict_data.extend(predict_keywords_count[0])
        return self.classifier.predict([predict_data])[0]


# ! Trainer Definition End

def run():
    aita = Aita()
    aita.main_trainer()
    pd = '''
    无人驾驶的公共汽车：可规划路线、自动售票、自动行驶。
    使用价值：各城市公交。现车辆的驾驶都需靠人来操作，在未来车辆驾驶可靠人工智能完成，这可避免人类身体素质疲劳、走神等缺点，优化车辆行驶性能。还可改变现在公交行驶路线短的缺点，增长行驶路线，缩短发车时间，可以得到更大的收益。在机器控制下的车辆有更精准的发车时间和到站时间，误差减小给居民生活带来更多便利。国家投入这种车辆的引进，今后的生活将更加简易。
    '''
    predict_res = aita.predict(pd)
    print("此回答的预估分数: " + str(predict_res) + " 分")

    pd2 = '''
    应用的想法：
    与安防结合，在监控系统中融入人脸识别、车辆分析等，提取有效信息。
    在公安系统中可以分析犯罪嫌疑人的线索，为警察办案提供有效的帮助。
    在社区门禁系统中也可以基于人脸识别来做人员甄别。
    商业价值：
    社区安保、公安、交通安全、学校安保等需要用到监控的领域都可以涉及。
    '''
    predict_res = aita.predict(pd2)
    print("此回答的预估分数: " + str(predict_res) + " 分")

    pd3 = '''
    AI产品：扫地机器人

    商业价值：
    扫地机器人是一个巨大、前景大好的市场，它占据智能家居场景，较有商业潜力。
    产品的消费群体大多数为中老年，一般情况下也是用于打扫卫生，而且随着人口老龄化趋势的增长，
    对智能家居的需求也会不断增加，加之年轻人对新兴AI产品的猎奇心理，该AI产品市场能保持平稳增长。
    而且扫地机器人的价值不仅在于让人们摆脱家庭清洁劳务，更是推动了智能家居的发展，
    使得越来越多的AI产品被研发出来，投入生产
    '''
    predict_res = aita.predict(pd3)
    print("此回答的预估分数: " + str(predict_res) + " 分")

    '''
    预估结果：
    9分
    8分
    6.5分
    '''


    #fp = open('./assets/answer.txt')
    #predict_answer = fp.readlines()
    #predict_res = aita.predict(''.join(predict_answer[2:]))
    #print("此回答的预估分数: " + str(predict_res[0]) + " 分")

if __name__ == '__main__':
    run()
    