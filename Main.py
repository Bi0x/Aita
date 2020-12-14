#! /usr/bin/python3
from WordAnalyser import WordAnalyser
from sklearn.ensemble import RandomForestRegressor
import re, math

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
        clf.fit(train_data, self.answers_score)
        self.classifier = clf
        print("------>>> 训练模型结束 <<<------")
    
    def predict(self, predict_answers):
        #print("------>>> 预测结果如下 <<<------")
        predict_keywords_count = self.get_keyword_counts(self.top_words, [predict_answers])
        predict_data = self.get_answers_len([predict_answers])
        predict_data.extend(predict_keywords_count[0])
        return self.classifier.predict([predict_data])[0]


# ! Trainer Definition End

def run():
    aita = Aita()
    aita.main_trainer()
    #predict_res = aita.predict('产品产品农村农村农村农村情况情况情况情况产品产品产品产品产品产品产品产品产品产品产品产品产品产品AIAIAIAIAIAIAIAIAIAIAIAIAIAIAI数据数据数据数据数据数据数据数据')
    #print(predict_res)
    predict_chunk = aita.file_reader("./assets/marked_answers2_simple.txt")
    predict_true_score = aita.get_total_score(predict_chunk)
    predict_main = aita.get_answers_main(predict_chunk)
    mean_diff = 0
    rss = 0
    for i in range(len(predict_main)):
        predict_res = aita.predict(predict_main[i])
        print("\n----------------------------")
        print("预测结果: " + str(predict_res))
        print("实际结果: " + str(predict_true_score[i]))
        print("----------------------------")
        rss += (predict_true_score[i] - predict_res) ** 2
        mean_diff += abs(predict_true_score[i] - predict_res)
    mse = rss / len(predict_main)
    print("平均误差: " + str(mean_diff / len(predict_main)))
    print("MSE: " + str(mse))

if __name__ == '__main__':
    run()
    