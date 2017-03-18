import os
import random
import time
import datetime
from matplotlib import pyplot as plt



def del_des(in_file, out_file, index):
  """ delete the index_th descriptor of in_file
  """
  infile = open(in_file, 'r')
  lines = infile.readlines()
  infile.close()

  outfile = open(out_file, 'w')
  for line in lines:
    line = line.split("\t")
    line = [line[x] for x in range(len(line)) if x != index]
    line = "\t".join(line)
    outfile.write(line)
  outfile.close()

def split_train(in_file, fold):
  """ split train data into 5 fold dataset,
      and merge 4 of 5 into a cv_train dataset, and leave 1 as cv_test dataset 
  """
  infile = open(in_file, 'r')
  lines = infile.readlines()
  infile.close()
  random.shuffle(lines)
  num = len(lines) / fold
  for i in range(fold):
    dir_name = "%d" % i
    if not os.path.exists(dir_name):
      os.mkdir(dir_name)
    outfile = open(dir_name + "/test.data", 'w')
    b = i * num 
    e = (i + 1) * num
    if i == fold - 1:
      e = len(lines)
    outfile.writelines(lines[b: e])
    outfile.close()

  for i in range(fold):
    in_str = " ".join(["%d/test.data" % j for j in range(fold) if j != i])
    out_str = "%d/train.data" % i
    os.system("cat %s > %s" % (in_str, out_str))



def tolibsvm(in_file, out_file):
  """
  get libsvm input format data
  """
  infile = open(in_file, 'r')
  lines = infile.readlines()
  infile.close()
  outfile = open(out_file, 'w')
  for line in lines:
    line = line[: -1].split("\t")
    n = []
    n.append(line[-1])
    for i, v in enumerate(line[: -1]):
      n.append("%d:%s" % (i + 1, v))
    line = "\t".join(n)
    outfile.write(line + "\n")
  outfile.close()
        


def get_best_para(in_file):
  """
  from grid.result get best c, g, p

  """
  infile = open(in_file, 'r')
  lines = infile.readlines()
  line = lines[-1]
  c, g, p, best_mse = line.split(" ")
  return c, g, p, best_mse





  

def data_2_dat(dir_name, index):
  if index < 0:
    os.system("cp %s/train.data %s/train.dat" % (dir_name, dir_name))
    os.system("cp %s/test.data %s/test.dat"  % (dir_name, dir_name))  
  else:
    del_des("%s/train.data" % dir_name, "%s/train.dat" % dir_name, index)
    del_des("%s/test.data" % dir_name, "%s/test.dat" % dir_name, index)



def delete_one(index, des_num, fold=5):

  def train_predict(dir_name, c, g, p):
    """"""
    tolibsvm(dir_name + "/test.dat", dir_name + "/libsvm_test.dat")   
    tolibsvm(dir_name + "/train.dat", dir_name + "/libsvm_train.dat")
    os.system("./svm-train -s 3 -t 2 -c %s -g %s -p %s %s/libsvm_train.dat %s/train.model > %s/train.result" % (c,g,p, dir_name, dir_name, dir_name))
    os.system("./svm-predict %s/libsvm_test.dat %s/train.model %s/test.result >> %s/predict_%d.out" % (dir_name, dir_name, dir_name, dir_name, des_num))

  # data_2_dat
  data_2_dat("master", index)

  ## master
  # convert data into libsvm input format
  tolibsvm("master/test.dat", "master/libsvm_test.dat")   
  tolibsvm("master/train.dat", "master/libsvm_train.dat")

  # find out best parameters
  os.system("python gridregression.py -log2c -1,5,1 -log2g -5,4,1 -log2p -5,0,1  -s 3 -t 2 -v 5 -svmtrain ./svm-train -gnuplot /usr/bin/gnuplot master/libsvm_train.dat >> master/grid.result")

  # output best para into file
  c, g, p, best_mse= get_best_para("master/grid.result")


  # use best para to train and predict master
  train_predict("master", c, g, p)

  ## 5 fold cross validation
  split_train("master/train.data", fold)
  for i in range(fold):
    data_2_dat(str(i), index)
    train_predict(str(i), c, g, p)

  return c, g, p, best_mse


def get_mse_r2_index(des_num, fold = 5):
  """
  get the min mse of the index or max R2 of the index
  """
  def get_mse_r2(dir_name):
    """
    get the mse and R2 of each model after the features were deleted one by one
    """
    infile = open(dir_name + "/predict_%d.out" % des_num, "r")
    mse_list = []
    r2_list = []
    for line in infile:
      if "Mean squared error" in line:
        mse = float(line.split(" ")[4])
        mse_list.append(mse)
      if "Squared correlation coefficient" in line:
        r2 = float(line.split(" ")[4])
        r2_list.append(r2)
    return mse_list, r2_list

  
  # get the master file (external test set) mse and R2
  m_mse, m_r2 = get_mse_r2("master")
  
  # get the fold file (5FCV) mse and R2 
  cv_mse = []
  cv_r2 = []
  for i in range(fold):
    mse_list, r2_list = get_mse_r2(str(i))
    cv_mse.append(mse_list)
    cv_r2.append(r2_list)
  
  # get the average mse and R2 of the fold
  cv_mse_ave = [0] * len(cv_mse[0])
  cv_r2_ave = [0] * len(cv_mse[0])
  for i in range(len(cv_mse[0])):
    for j in range(len(cv_mse)):
      cv_mse_ave[i] += cv_mse[j][i]
      cv_r2_ave[i] += cv_r2[j][i]
    cv_mse_ave[i] = cv_mse_ave[i] / len(cv_mse)
    cv_r2_ave[i] = cv_r2_ave[i] / len(cv_mse)

  assert len(m_mse) == len(cv_mse_ave), "something error"

  # define the target function to optimization(maxmal R2 or minmal MSE), the function is r2 or mse of cross validation or test set validation
  weight = 0
  mse = [0] * len(m_mse)
  r2 = [0] * len(m_r2)
  for i in range(len(m_mse)):
    mse[i] = weight * m_mse[i] + (1 - weight) * cv_mse_ave[i]
    r2[i] = weight * m_r2[i] + (1 - weight) * cv_r2_ave[i]

  mse_min_index = mse.index(sorted(mse)[0])
  r2_max_index = r2.index(sorted(r2, reverse=True)[0])

  if des_num == special_num:
    return mse_min_index, r2_max_index, cv_mse_ave[r2_max_index], cv_r2_ave[r2_max_index], m_mse[r2_max_index], m_r2[r2_max_index]

  # if the index of max R2 or min mse is 0, then we will remove the second sequence index of descriptors
  if mse_min_index == 0:
    mse_min_index = mse.index(sorted(mse)[1])
  if r2_max_index == 0:
    r2_max_index = r2.index(sorted(r2, reverse=True)[1])

  return mse_min_index, r2_max_index, cv_mse_ave[r2_max_index], cv_r2_ave[r2_max_index], m_mse[r2_max_index], m_r2[r2_max_index]


def analysis_plot(index_list, cv_mse_ave_list, cv_r2_ave_list, m_mse_list, m_r2_list):
  unde_str=["u"]
  for i in range(len(index_list)-1, -1, -1):
    unde_str.insert(index_list[i]-1, i+1)
  with open('deleted.list', 'w') as f:
    s = [str(x) + "\t" for x in unde_str]
    f.writelines(s) 

  x = range(len(cv_mse_ave_list)-1, -1, -1)

#  plt.plot(x, cv_mse_ave_list, "r>--")
  plt.plot(x, cv_r2_ave_list, "bo-")
#  plt.plot(x, m_mse_list, "k>--")
  plt.plot(x, m_r2_list, "go-")
  plt.xlabel("Number of features selected")
  plt.ylabel("R_Squared")
  plt.show()



if __name__ == "__main__":

  # pre-define parameters 
  fold = 5
  des_num = 24 #set number of features to elimination

  
  


  index_list = []
  cv_mse_ave_list = []
  cv_r2_ave_list = []
  m_mse_list = []
  m_r2_list = []



  # delete zero descriptor
  special_num = des_num + 1 
  c, g, p, best_mse = delete_one(-1, special_num)
  _, r2_max_index, cv_mse_ave, cv_r2_ave, m_mse, m_r2 = get_mse_r2_index(special_num)


  #index_list.append(r2_max_index)
  cv_mse_ave_list.append(cv_mse_ave)
  cv_r2_ave_list.append(cv_r2_ave)
  m_mse_list.append(m_mse)
  m_r2_list.append(m_r2)




  # traverse descriptor

  while True:
    t0 = time.time()
    
    if des_num < 1:
      break

    infile = open("master/Best_%d.para" % des_num, 'w')

    # delete zero descriptor 
    c, g, p, best_mse = delete_one(-1, des_num)
    infile.write("SFE\t0\t%s\t%s\t%s\t%s" % (c, g, p, best_mse))

    # delete descriptors one by one and   
    for i in range(des_num):
      c, g, p, best_mse = delete_one(i, des_num)
      infile.write("SFE\t%d\t%s\t%s\t%s\t%s" % (i+1, c, g, p, best_mse))

    infile.close()

    #os.system("rm libsvm_train.dat.out libsvm_train.data.out")
    
    # get the best result after delection the i(index) descriptor 
    _, r2_max_index, cv_mse_ave, cv_r2_ave, m_mse, m_r2 = get_mse_r2_index(des_num, fold=5)
    
    index_list.append(r2_max_index)
    cv_mse_ave_list.append(cv_mse_ave)
    cv_r2_ave_list.append(cv_r2_ave)
    m_mse_list.append(m_mse)
    m_r2_list.append(m_r2)

    # delete the index_th descriptor according to r2 
    del_des("master/train.data", "master/train.data", r2_max_index-1)
    del_des("master/test.data", "master/test.data", r2_max_index-1)

    t1 = time.time()
    print("current des num: %d, duration: %.3f" % (des_num, t1-t0))
  
    des_num -= 1
    

  analysis_plot(index_list, cv_mse_ave_list, cv_r2_ave_list, m_mse_list, m_r2_list)
  with open("result.list", 'w') as result:
    for i in range(len(cv_mse_ave_list)):
      result.write(cv_mse_ave_list[i]+"\t"+cv_r2_ave_list[i]+"\t"+m_mse_list[i]+"\t"+m_r2_list[i]+"\n")
    




