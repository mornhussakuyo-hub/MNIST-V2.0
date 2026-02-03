import numpy as np
import pandas as pd
def load_data(train_path,test_path):
    #从文件加载数据
    print("Loading data...")
    print(f"Reading train_data from {train_path}")
    train_df=pd.read_csv(train_path)
    #用iloc对数据分割
    train_labels=train_df.iloc[:,0].values
    train_images=train_df.iloc[:,1:].values
    print(f"Reading test_data from {test_path}")
    test_df=pd.read_csv(test_path)
    test_labels=test_df.iloc[:,0].values
    test_images=test_df.iloc[:,1:].values
    print(f"Train data shape: {train_images.shape}")
    print(f"Test data shape: {test_images.shape}")
    print(f"Train label shape: {train_labels.shape}")
    print(f"Test label shape: {test_labels.shape}")
    return train_images,train_labels,test_images,test_labels

def normalize_data(images,data_name):
    print(f"Normalizing {data_name}...")
    #向量化操作归一化数据
    return images/255.0

def one_hot_encode(labels,num_classes=10):
    print("Encoding labels to one-hot...")
    #将标签转化为one-hot编码
    num_samples=labels.shape[0]
    #检查labels类型
    if labels.dtype.kind not in ("i","u"):
        print(f"WARNING: passing labels type is {labels.dtype}, transforming it to integer..")
        labels=labels.astype(np.int32)
    #创建零矩阵
    one_hot=np.zeros((num_samples,num_classes)) 
    #智能索引
    one_hot[np.arange(num_samples),labels]=1 
    print("Encoding Done.")
    return one_hot

def split_train_val(images,labels,val_radio=0.2,usage_radio=1):
    #分割训练验证集
    #获取总个数
    num_samples=images.shape[0]
    #使用部分数据
    usage_num=int(num_samples*usage_radio )
    all_indices=np.arange(num_samples)
    usage_indices=np.random.choice(all_indices,size=usage_num,replace=False)
    images=images[usage_indices]
    labels=labels[usage_indices]
    #更新总个数
    num_samples=usage_num
    print(f"Using {usage_num}/{len(all_indices)} samples, usage radio: {usage_radio}")
    print(f"Spliting data using valid radio: {val_radio}")
    #打乱数据
    indices=np.random.permutation(num_samples)
    split_idx=int(num_samples*(1-val_radio))
    #分割数据
    train_indices=indices[:split_idx]
    val_indices=indices[split_idx:]
    train_images=images[train_indices]
    train_labels=labels[train_indices]
    val_images=images[val_indices]
    val_labels=labels[val_indices]
    print(f"Train images shape: {train_images.shape}") 
    print(f"Train labels shape: {train_labels.shape}") 
    print(f"Valid images shape: {val_images.shape}") 
    print(f"Valid labels shape: {val_labels.shape}") 
    print("Split data done.")
    return train_images,train_labels,val_images,val_labels

if __name__ == "__main__":

    pass
