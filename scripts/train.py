import argparse
import numpy as np
import sys
import os
from pathlib import Path


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import TwoLayerNet
from utils import load_data,normalize_data,one_hot_encode,split_train_val
from common import validate_file_path,safe_path_resolution

def parse_args():
    parser=argparse.ArgumentParser(description="MNIST V2.0\nAttention: all data must be in .csv format.")

    parser.add_argument("--traindt",type=str,default="../data/mnist_train.csv",help="Path to train data CSV file (default: ../data/mnist_train.csv)")
    parser.add_argument("--testdt",type=str,default="../data/mnist_test.csv",help="Path to test data CSV file (default: ../data/mnist_test.csv)")
    parser.add_argument("--model-name",type=str,required=True,help="Model name (required)")
    parser.add_argument("--model-dir",type=str,default="../models",help="Model save directory (default: ../models)")
    parser.add_argument("--results-dir",type=str,default="../results",help="Results save directory (default: ../results)")
    parser.add_argument("--LR-dc",type=str,default="off",help="Decreasing LR with time (default: off)")
    parser.add_argument("--LR-dc-num",type=int,default=20,help="Number of epochs between each LR decrease (default: 20)")
    parser.add_argument("--epochs",type=int,default=50,help="Train epochs (default: 50)")
    parser.add_argument("--batch-size",type=int,default=32,help="Batch size (default: 32)")
    parser.add_argument("--learning-rate",type=float,default=0.01,help="Learning-rate (default: 0.01)")
    parser.add_argument("--hidden-size",type=int,default=128,help="Hidden-layer size (default: 128)")
    parser.add_argument("--val-ratio",type=float,default=0.2,help="Proportion for valid-set (default: 0.2)")
    parser.add_argument("--usage-ratio",type=float,default=1.0,help="Proportion used for the training-set (default: 1.0)")
    parser.add_argument("--reg-rate",type=float,default=0.0,help="Regularization rate (default: 0.0)")
    parser.add_argument("--no-save", action="store_true",help="Not to save model")
    return parser.parse_args()

def main():
    args=parse_args()
    input_size=784
    hidden_size=args.hidden_size
    output_size=10
    learning_rate=args.learning_rate
    num_epochs=args.epochs
    batch_size=args.batch_size
    val_ratio=args.val_ratio
    data_usage_ratio=args.usage_ratio
    reg_rate=args.reg_rate
    learning_rate_decrease=args.LR_dc
    learning_rate_decrease_time=args.LR_dc_num


    print("\nProcess start...")
    print(f"Network: {input_size} -> {hidden_size} -> {output_size}")
    print(f"Hypermarameter: LR={learning_rate}, Epochs={num_epochs}, Batch={batch_size}")
    print(f"Learning rate decrease was {learning_rate_decrease}.")
    print(f"Regularization rate was {reg_rate}.")

    print("\nStart loading data.")

    try: 
        train_path=safe_path_resolution(args.traindt)
        test_path=safe_path_resolution(args.testdt)
        validate_file_path(train_path)
        validate_file_path(test_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check if the path was right")
        exit(1)

    train_images,train_labels,test_images,test_labels=load_data(train_path,test_path)
    print("Load data done.")

    print("\nStart data pre-processing.")
    train_images=normalize_data(train_images,"train_images")
    test_images=normalize_data(test_images,"test_images")
    
    train_images,train_labels,val_images,val_labels=split_train_val(train_images,train_labels,val_ratio,data_usage_ratio)
    train_labels=one_hot_encode(train_labels)
    val_labels=one_hot_encode(val_labels)
    test_labels=one_hot_encode(test_labels)
    print("Data pre-processing done.")

    print("\nStart initializing model.")
    model=TwoLayerNet(input_size,hidden_size,output_size,"he",args.model_name)
    print("Initializing model done.")

    print("\nStart train.")
    num_train=train_images.shape[0]
    num_batches=num_train//batch_size

    train_loss_history=[]
    val_loss_history=[]
    train_acc_history=[]
    val_acc_history=[]

    for epoch in range(num_epochs):
        indices=np.random.permutation(num_train)
        train_images_shuffled=train_images[indices]
        train_labels_shuffled=train_labels[indices]

        epoch_loss=0.0

        for batch_idx in range(num_batches):
            start=batch_idx*batch_size
            end=start+batch_size
            X_batch=train_images_shuffled[start:end]
            y_batch=train_labels_shuffled[start:end]
            
            y_hat=model.forward(X_batch)
            batch_loss=model.compute_loss(y_hat,y_batch)
            epoch_loss+=batch_loss

            gradients=model.backward(X_batch,y_batch,y_hat,reg_rate)
            model.update_parameters(gradients,learning_rate)
        avg_train_loss=epoch_loss/num_batches

        train_acc=model.accuracy(train_images,train_labels)
        val_acc=model.accuracy(val_images,val_labels)

        val_y_hat=model.forward(val_images)
        val_loss=model.compute_loss(val_y_hat,val_labels)

        train_loss_history.append(avg_train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        print(f"Epoch: {epoch+1:3d}/{num_epochs} | Loss: {avg_train_loss:.4f} (train)/{val_loss:.4f} (valid) | Acc: {train_acc*100:.2f}% (train)/{val_acc*100:.2f}% (valid)")
        
        if learning_rate_decrease=="on":
            if (epoch+1)%learning_rate_decrease_time==0 :
                learning_rate/=2
                print(f"Decreasing LR to {learning_rate}")
    
    print("\nRunning on test data...")
    test_acc=model.accuracy(test_images,test_labels)
    print(f"Test data acc: {test_acc*100:.2f}%")

    if not args.no_save:
        print("\nSaving model...")
        results_dir=args.results_dir
        models_dir=args.model_dir
        
        os.makedirs(results_dir,exist_ok=True)
        os.makedirs(models_dir,exist_ok=True)


        results_path=os.path.join(results_dir,f"{args.model_name}_training_history.npz")
        np.savez(results_path,train_loss=train_loss_history,val_loss=val_loss_history,train_acc=train_acc_history,val_acc=val_acc_history)
        print(f"Training history has been saved at: {results_path}")
        
        model_path=os.path.join(models_dir,f"{args.model_name}.npz")
        model.save(model_path)
        print(f"Model has been saved at: {model_path}")
    else:
        print("No Saving was on.")

if __name__=="__main__":
    main()
    pass