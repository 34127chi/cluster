import pdb
import numpy as np
import re
from bs4 import BeautifulSoup
from ch_tradi2simli.langconv import Converter
tradi2sim_obj = Converter('zh-hans')

def read_vectors(path, topn = 0):  # read top n word vectors, i.e. top is 10000
    lines_num, dim = 0, 0
    vectors = {}
    iw = []
    wi = {}
    with open(path, encoding='utf-8', errors='ignore') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                dim = int(line.rstrip().split()[1])
                continue
            lines_num += 1
            tokens = line.rstrip().split(' ')
            vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
            iw.append(tokens[0])
            if topn != 0 and lines_num >= topn:
                break
    for i, w in enumerate(iw):
        wi[w] = i
    return vectors, iw, wi, dim

def batch_iter(train_x, batch_size):
    batch_size = batch_size 
    data_len = len(train_x)
    num_batch = int((data_len - 1) / batch_size) + 1
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield train_x[start_id:end_id]

def strQ2B(line):
    '''
    desc:
        全角转半角

    param[in]:
        line:句子

    return:
        q2b_str:转换后的句子
    '''
    q2b_str = ''
    for char in line:
        inside_code = ord(char)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
        if inside_code < 0x0020 or inside_code > 0x7e:
            q2b_str += char
            continue
        q2b_str += chr(inside_code)
    
    return q2b_str

def preprocess(line):
    '''
    desc:
        预处理

    param[in]:
        line:句子

    return:
        line:预处理后的句子
    '''
    line = strQ2B(line)#全角转半角
    line = line.lower()#大小写转换
    soup = BeautifulSoup(line, 'lxml')
    line = soup.get_text()#去除数据中非文本部分 html 
    line = line.replace('\r', '').replace('\n','')#去除额外的符号
    #line = line.replace('\r', '').replace('\n','').replace(' ', '')#去除额外的符号
    #b = re.compile('[^\u4e00-\u9fa5a-z+.,?，。？!℃\-]', re.I)
    b = re.compile('[^\u4e00-\u9fa5a-z0-9+.,?，。？!℃\-[] ]', re.I)
    line = b.sub('', line)
    b = re.compile('\[\w+\]', re.I)#表情去除
    line = b.sub('', line)

    if line.isdigit():
        return ""
    
    line = tradi2sim_obj.convert(line)

    return line

def read_file(filename):
    content = []
    with open(filename, 'r') as file_r:
        for line in file_r:
            line = preprocess(line)
            if line == '':
                continue
            content.append(line)
    return content 
