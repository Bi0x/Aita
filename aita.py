#! /usr/bin/python3

from WordAnalyser import WordAnalyser
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier
import joblib
import re
import argparse
import sklearn.svm as svm
import numpy as np

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
        self.answers_chunk = self.file_reader_withscore(
            './assets/marked_answers.txt')
        self.answers_main = self.get_answers_main(self.answers_chunk)
        self.top_words = word_analyser.keyword_analyse(self.answers_main)
        self.classifier = None

    def show_question_name(self):
        # ? 显示问题
        fp = open('./assets/question.txt')
        print("针对的问题：《" + fp.readline()[:-1] + "》")
        fp.close()

    def file_reader_withscore(self, path):
        # ? 读取答案数据，并将答案分组
        train_file = open(path)
        file_text = "".join(train_file.readlines())
        regex_pattern = r'((\d{12}.*?#Total: *?\d\.*\d*/10))'
        reg = re.compile(regex_pattern, re.DOTALL)
        return reg.findall(file_text)

    def file_reader(self, path):
        # ? 读取答案数据，并将答案分组(不包括成绩信息)
        train_file = open(path)
        file_text = "".join(train_file.readlines())
        regex_pattern = r'(\d{12}([^@#]*)?)'
        reg = re.compile(regex_pattern, re.DOTALL)
        return reg.findall(file_text)

    def get_total_score(self, answers):
        # ? 分析答案，获取评分
        total_score = []
        regex_pattern = r'#Total: *?(\d\.*\d*)/10'
        reg = re.compile(regex_pattern, re.DOTALL)
        for answer in answers:
            total_score.append(float(reg.findall(answer[0])[0]))
        return total_score

    def get_answers_main(self, answers):
        # ? 获取主要回答内容
        answers_main = []
        regex_pattern = r':\d\d\n([^@#]*)?'
        reg = re.compile(regex_pattern, re.DOTALL)
        for answer in answers:
            answers_main.append(reg.findall(answer[0])[0])
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
        self.answers_score = self.get_total_score(self.answers_chunk)
        self.answers_len = self.get_answers_len(self.answers_main)
        self.answers_keywordcount = self.get_keyword_counts(
            self.top_words, self.answers_main)
        train_data = []

        for i in range(len(self.answers_main)):
            per_line = [self.answers_len[i]]
            per_line.extend(self.answers_keywordcount[i])
            train_data.append(per_line)
        print("------>>> 正在训练模型 <<<------")
        clf = RandomForestRegressor(
            n_estimators=100,       # 子树数量
            max_depth=8,            # 最大深度
            min_samples_leaf=6,     # 叶节点最小样本数
            max_features=7,         # 节点分裂时参与判断的最大特征数
            oob_score=True,         # 使用袋外样品进行估算泛化精度
            random_state=10         # 随机数种子
        )
        #clf= svm.NuSVR(nu = 0.5)
        clf.fit(train_data, self.answers_score)
        # ? 这一块是处理重要性的
        variable_importance = [('answer.length', clf.feature_importances_[0])] # answer.length is the first predictor variable
        for i in range(len(self.top_words)):
            variable_importance.append((self.top_words[i][0], clf.feature_importances_[i+1]))
        variable_importance.sort(key=lambda x: x[1], reverse=True)
        for i in range(len(variable_importance)):
            print("预测变量：" + variable_importance[i][0] + "\t\t\t重要性: %4.3f" % (variable_importance[i][1]))
        # ? 保存模型
        self.classifier = clf
        print("----------------------------")
        print("------>>> 训练模型结束 <<<------")

    def predict(self, predict_answers):
        #print("------>>> 预测结果如下 <<<------")
        predict_keywords_count = self.get_keyword_counts(
            self.top_words, [predict_answers])
        predict_data = self.get_answers_len([predict_answers])
        predict_data.extend(predict_keywords_count[0])
        return self.classifier.predict([predict_data])[0]


# ! Trainer Definition End

def test():
    # ? 测试模型，使用自带的回答
    aita = Aita()
    aita.classifier = joblib.load('Aita.model')
    predict_chunk = aita.file_reader_withscore(
        "assets/marked_answers2_simple.txt")
    predict_true_score = aita.get_total_score(predict_chunk)
    predict_main = aita.get_answers_main(predict_chunk)
    total_absolute_diff = 0
    rss = 0
    residuals = []
    pred_val = [] # save predicted values for computing quantiles
    print('No\tTrue value\tPredicted Value\tDifference')
    for i in range(len(predict_main)):
        predict_res = aita.predict(predict_main[i])
        print('%d\t%4.2f\t%4.2f\t%4.2f' % (i+1, predict_true_score[i], predict_res, predict_res-predict_true_score[i]))
        residuals.append((predict_true_score[i] - predict_res))
        rss += (predict_true_score[i] - predict_res) ** 2
        total_absolute_diff += abs(predict_true_score[i] - predict_res)
        pred_val.append(predict_res)
    mse = rss / len(predict_main)
    print('---------------------------------------------------------------')
    print('平均绝对误差: %4.5f' % (total_absolute_diff / len(predict_main)))
    print('Mean Squared Error (MSE): %4.5f' % (mse))
    print(' 5th quantile: %4.2f' % (np.quantile(pred_val, 0.05)))    
    print('25th quantile: %4.2f' % (np.quantile(pred_val, 0.25)))
    print('50th quantile: %4.2f' % (np.quantile(pred_val, 0.50)))
    print('75th quantile: %4.2f' % (np.quantile(pred_val, 0.75)))    
    print('95th quantile: %4.2f' % (np.quantile(pred_val, 0.95)))
    
    # 下面都是画图的
    '''
    figure, axes=plt.subplots() #得到画板、轴
    axes.boxplot(residuals, patch_artist=True) #描点上色
    plt.show()
    
    plt.plot([x for x in range(len(predict_main))], residuals, marker='o', color='red')
    plt.rcParams['font.sans-serif'] = ['PingFang HK']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title("预测结果",fontsize=14)
    plt.xlabel("回答编号",fontsize=14)
    plt.ylabel("误差^2", fontsize=14)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
    plt.gcf().set_size_inches(15, 7)
    plt.show()
    '''


def train():
    # ? 训练模型
    aita = Aita()
    aita.main_trainer()
    print("------>>> 正在保存模型 <<<------")
    joblib.dump(aita.classifier, 'Aita.model')
    fp = open('./assets/ModelParameters.txt', 'w')
    fp.write("max_depth=10\nn_estimators=1000\nmin_samples_split=2")
    fp.close()
    print("------>>> 保存模型结束 <<<------")


def run(path):
    # ? 预测数据
    aita = Aita()
    aita.classifier = joblib.load('Aita.model')
    predict_chunk = aita.file_reader(path)
    predict_main = aita.get_answers_main(predict_chunk)
    for i in range(len(predict_main)):
        predict_res = aita.predict(predict_main[i])
        print("\n----------------------------")
        print("预测结果: " + str(predict_res))
        print("----------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", help="训练模型", action="store_true")
    parser.add_argument("-predict", help="-predict filename.txt")
    parser.add_argument("-test", help="测试模型", action="store_true")

    args = parser.parse_args()

    if args.train:
        train()
    elif args.predict:
        run(args.predict)
    elif args.test:
        test()
    else:
        train()
        test()
    # run()
