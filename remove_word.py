#! /usr/bin/python3

def read_remove_word():
    # ? 获取需要剔除的字符和词语
    fp = open('./assets/remove_words.txt', 'r')
    return ''.join(fp.readlines()).split('|')

def create_remove_words_file():
    remove_words = [
        u'的',     u'，',   u'和',      u'是',      u'随着', 
        u'对于',   u'对',   u'等',      u'能',      u'都', 
        u'。',     u' ',    u'、',      u'中',      u'在', 
        u'了',     u'通常', u'如果',    u'我们',    u'2',
        u'需要',   u'\n',   u'.',       u'：',      u'，',
        u',',      u'（',   u'）',      u'与',      u'有', 
        u'会',     u'也',   u'以及',    u'可',      u'通过', 
        u'上',     u'可以', u'并',      u"\u3000",  u'1',
        u'to',     u'\t',   u'一个',    u'将',      u'到',
        u'“',      u'”',    u'不',      u'地',      u'in',
        u'于',     u'还',   u'我',      u'人',      u'为',
        u'更',
    ]
    fp = open('./assets/remove_words.txt', 'w')
    for i in remove_words:
        fp.write(i + "|")
    fp.close()

if __name__ == '__main__':
    create_remove_words_file()