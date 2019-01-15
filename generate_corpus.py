import pandas as pd
import pdb
import sys
from util import  * 

def get_dialog(filename, total_filtered_set = set([])):
    df =  pd.read_csv(filename)
    print(df.head())
    print(df.userid.unique().shape)

    total_filtered_set = set(['傻逼','日你', '妈的', 'tmd', '你妈逼', '操你妈', '草你妈', '艹你妈', '草泥马', '傻子', '傻逼', '撒比', '他妈的', '傻逼', '草泥马', '骗子', '麻痹'])#用于敏感词过滤

    count_number = 0

    with open(filename + '.conv', 'w') as file_conv_w, open(filename + '.question', 'w') as file_que_w, open(filename + '.ans', 'w') as file_ans_w:
        for userid in df.userid.unique():#根据用户id筛选问答对
            tmp_conv = df.loc[df.userid == userid] 
            sorted_tmp_conv = tmp_conv.sort_values(by=['creationdate'])#根据时间排序

            user_q = ""
            customer_a = ""
            user_flag = 1#用户标志
            customer_flag = 2#客服标志
            has_user_q = False#含有用户文本
            has_customer_a = False#含有客服文本
            is_pair = False#是否形成问答对
            text_flag = 1#消息类型

            for idx, row in sorted_tmp_conv.iterrows():
                count_number = count_number + 1
                if count_number % 10000 == 1:
                    print("count number:" + str(count_number))
                #print(row["messagetype"])
                if row["messagetype"] != text_flag:
                    continue
                if is_pair == True and row['direction'] == user_flag:
                    #敏感词过滤
                    has_filtered_word = False
                    for filtered_word in total_filtered_set:
                        if filtered_word in user_q or filtered_word in customer_a:
                            print(filtered_word)
                            has_filtered_word = True
                            user_q = ""
                            customer_a = ""
                            is_pair = False
                            has_user_q = False
                            has_customer_a = False
                            break
                    if has_filtered_word == True:#如果有敏感文本就剔除
                        continue
                    if len(user_q) > 5:
                        file_conv_w.write(user_q + '\t' + customer_a + '\n')
                        file_que_w.write(user_q + '\n')
                        file_ans_w.write(customer_a + '\n')
                    user_q = ""
                    customer_a = ""
                    is_pair = False
                    has_user_q = False
                    has_customer_a = False
                if row['direction'] == user_flag:
                    tmp_str = preprocess(str(row['content']))
                    if tmp_str == '':
                        continue
                    user_q = user_q + " " + tmp_str 
                    has_user_q = True
                elif row['direction'] == customer_flag:
                    tmp_str = preprocess(str(row['content']))
                    if tmp_str == '':
                        continue
                    customer_a = customer_a + " " + tmp_str 
                    has_customer_a = True
                else:
                    if has_user_q == True and has_customer_a == True:
                        #敏感词过滤
                        for filtered_word in total_filtered_set:
                            if filtered_word in user_q or filtered_word in customer_a:
                                has_filtered_word = True
                                user_q = ""
                                customer_a = ""
                                is_pair = False
                                has_user_q = False
                                has_customer_a = False
                                break
                        if has_filtered_word == True:
                            continue

                        if len(user_q) > 5:
                            file_conv_w.write(user_q + '\t' + customer_a + '\n')
                            file_que_w.write(user_q + '\n')
                            file_ans_w.write(customer_a + '\n')
                                
                        user_q = ""
                        customer_a = ""
                        is_pair = False
                        has_user_q = False
                        has_customer_a = False

                if has_user_q == True and has_customer_a == True:#如果用户文本和客服文本均有，那么形成问答对
                    is_pair = True

    pass

if __name__ == '__main__':
    '''
    基于问答对的形式构建用户语料库集
    '''
    sensitive_data_dir = './sensitive_data/'#敏感词
    for sensitive_filename in os.listdir(sensitive_data_dir):
        sensitive_set = set(read_file(os.path.join(sensitive_data_dir, sensitive_filename)))

    filename = './data/train.csv'#线上用户与客服的问答对
    get_dialog(filename, sensitive_set)
    pass
