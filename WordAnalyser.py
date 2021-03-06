#! /usr/bin/python3
import csv
import jieba
import collections
import logging

jieba.setLogLevel(logging.INFO)


class WordAnalyser:

    def read_remove_word(self):
        # ? 获取需要剔除的字符和词语
        self.create_remove_words()
        fp = open('./assets/remove_words.csv', 'r')
        csv_loader = csv.reader(fp)
        return [eval(i).decode('utf-8') for i in list(csv_loader)[0]]

    def create_remove_words(self):
        # ? 添加需要剔除的关键词
        #remove_words = ['没有词会被删除']
        # ? 不是字的
        remove_words = [
            u'\t',     u'，',     u'。',     u' ',      u'、',
            u'了',     u'\n',     u'.',      u'：',     u'，',
            u',',      u'（',     u'）',     u"\u3000", u'“',
            u'”',      u'？',
        ]
        # ? 是字的
        # remove_words.extend([
        #     u'的',          u'一个',        u'将',          u'到',      u'不',
        #     u'地',          u'于',          u'还',          u'我',      u'人',
        #     u'为',          u'更',          u'就',          u'Marking', u'Scheme',
        #     u'Writing',     u'Feasibility', u'Creativity',  u'中',      u'在',
        #     u'Potential',   u'impact',      u'Total',       u'就',      u'与',
        #     u'AI',          u'English',     u'上',          u'通过',    u'有',
        #     u'你',          u'可以',        u'通常',        u'如果',    u'我们',
        #     u'并',          u'需要',        u'和',          u'是',      u'随着',
        #     u'对于',        u'对',          u'等',          u'能',      u'都',
        #     u'会',          u'也',          u'以及',        u'可',      u'to'
        # ])
        # 是数字的
        # remove_words.extend([
        #     u'4',    u'3',       u'1.5',     u'1',        u'10'
        # ])
        # ? 测试
        # remove_words.extend([
        #     u'的',      u'就',      u'可以',        u'以及',
        #     u'会',      u'也',      u'和',          u'都',
        #     u'与',      u'可',      u'在',          u'中'
        # ])
        fp = open('./assets/remove_words.csv', 'w')
        csv_saver = csv.writer(fp)
        encode_remove_words = []
        for i in remove_words:
            encode_remove_words.append(i.encode('utf-8'))
        csv_saver.writerow(encode_remove_words)
        fp.close()

    def keyword_analyse(self, answers):
        # ? 关键词分析
        remove_words = self.read_remove_word()
        word_objects = []
        for answer in answers:
            word_list = jieba.cut_for_search(answer) # for jieba useage, check https://github.com/fxsjy/jieba
            for i in word_list:
                if i not in remove_words:
                    word_objects.append(i)
        word_counts = collections.Counter(word_objects)
        word_counts_tops = word_counts.most_common(30)
        # print(word_counts_tops)
        return word_counts_tops


if __name__ == '__main__':
    word_analyser = WordAnalyser()
    print("------->>> 目前剔除的词如下: <<<-------")
    print(word_analyser.read_remove_word())
