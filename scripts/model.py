import numpy as np

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init="he",model_name=""):
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.model_name=model_name
        #初始化权重
        if weight_init=="he":
            #He初始化，推荐ReLU
            print("initializing weights using HE method...")
            self.W1=np.random.randn(hidden_size,input_size)*np.sqrt(2.0/input_size)
            self.W2=np.random.randn(output_size,hidden_size)*np.sqrt(2.0/hidden_size)
        elif weight_init=="xavier":
            #Xavier初始化
            print("initializing weights using Xavier method...")
            self.W1=np.random.randn(hidden_size,input_size)*np.sqrt(1.0/input_size)
            self.W2=np.random.randn(output_size,hidden_size)*np.sqrt(1.0/hidden_size)
        elif weight_init=="normal":
            #简单随机
            print("initializing weights using normal random...")
            self.W1=0.01*np.random.randn(hidden_size,input_size)
            self.W2=0.01*np.random.randn(output_size,hidden_size)
        #初始化偏置
        print("initializing bias...")
        self.b1=np.zeros(hidden_size)
        self.b2=np.zeros(output_size)
        #中间结果存储
        self.cache={}

    def get_model_name(self):
        return self.model_name

    def relu(self,x):
        #ReLU激活函数
        return np.maximum(0,x)
    
    def d_relu(self,x):
        #ReLU导数
        return (x>0).astype(np.float32)
    
    def softmax(self,x):
        #softmax函数
        if x.ndim==1:
            x=x.reshape(1,-1)
            exp_x=np.exp(x-np.max(x,axis=1,keepdims=True))
            return (exp_x/np.sum(exp_x,axis=1,keepdims=True)[0]).reshape(-1)

        exp_x=np.exp(x-np.max(x,axis=1,keepdims=True))
        return exp_x/np.sum(exp_x,axis=1,keepdims=True)
    
    def forward(self,X):
        #输入层
        self.cache["X"]=X
        self.cache["z1"]=X@self.W1.T+self.b1
        #过激活函数
        self.cache["a1"]=self.relu(self.cache["z1"])
        #隐藏层
        self.cache["z2"]=self.cache["a1"]@self.W2.T+self.b2
        #过softmax
        self.cache["y_hat"]=self.softmax(self.cache["z2"])
        return self.cache["y_hat"]
    def compute_loss(self,y_hat,y):
        #输出数
        num_samples=y.shape[0]
        #clip小量防爆NaN
        epsilon=1e-8
        y_hat=np.clip(y_hat,epsilon,1.-epsilon)
        #计算交叉熵损失
        loss=-np.sum(y*np.log(y_hat))/num_samples
        return loss
    def backward(self,X,y,y_hat,reg_lambda=0.0):
        num_samples=X.shape[0]
        #输出层梯度
        dz2=y_hat-y
        dW2=(dz2.T@self.cache["a1"])/num_samples+reg_lambda*self.W2
        db2=np.sum(dz2,axis=0)/num_samples
        #隐藏层梯度
        da1=dz2@self.W2
        dz1=da1*self.d_relu(self.cache["z1"])
        dW1=dz1.T@X/num_samples+reg_lambda*self.W1
        db1=np.sum(dz1,axis=0)/num_samples

        #把梯度打包到字典
        gradients={
            "dW1":dW1,
            "db1":db1,
            "dW2":dW2,
            "db2":db2
        }
        return gradients
    
    def update_parameters(self,gradients,learning_rate):
        #更新参数
        self.W1-=learning_rate*gradients["dW1"]
        self.b1-=learning_rate*gradients["db1"]
        self.W2-=learning_rate*gradients["dW2"]
        self.b2-=learning_rate*gradients["db2"]

    def predict(self,X):
        #输入数据集进行预测
        y_hat=self.forward(X)
        #argmax函数可以获取最大值的索引
        return np.argmax(y_hat,axis=1)
    
    def accuracy(self,X,y):
        #计算acc
        predictions=self.predict(X)
        true_labels=np.argmax(y,axis=1)
        #通过向量化操作，获得布尔变量组成的向量，然后求平均值就是准确率
        accuracy=np.mean(predictions==true_labels)
        return accuracy

    def save(self,filepath):
        #保存模型
        np.savez(filepath,W1=self.W1,b1=self.b1,W2=self.W2,b2=self.b2,model_name=self.model_name)

    def load(self,filepath):
        #加载模型
        data=np.load(filepath)
        self.W1=data["W1"]
        self.b1=data["b1"]
        self.W2=data["W2"]
        self.b2=data["b2"]
        self.model_name=data["model_name"] if "model_name" in data else ""

