import torch
import torch.nn as nn
from utils import *
from models.ENet import ENet
import sys
from tqdm import tqdm

def train(opt):
    # Defining the hyperparameters
    device =  opt.device
    rh = opt.resize_height
    rw = opt.resize_width
    batch_size = opt.batch_size
    epochs = opt.epochs
    lr = opt.learning_rate
    print_every = opt.print_every
    eval_every = opt.eval_every
    save_every = opt.save_every
    num_classes = opt.num_classes
    wd = opt.weight_decay
    ip = opt.input_path_train
    lp = opt.label_path_train
    ipv = opt.input_path_val
    lpv = opt.label_path_val
    optim = opt.optimizer
    save_path = opt.save_model_path
    if os.path.exists(save_path):
        pass
    else:
        os.makedirs(save_path, exist_ok=True)
    print ('[INFO]Defined all the hyperparameters successfully!')
    
    # Get the class weights
    print ('[INFO]Starting to define the class weights...')
    pipe = loader(ip, lp, batch_size='all')
    class_weights = get_class_weights(pipe, num_classes)
    print ('[INFO]Fetched all class weights successfully!')

    # Get an instance of the model
    enet = ENet(num_classes).to(device)
    print ('[INFO]Model Instantiated!')

    # Define the criterion and the optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    if optim == 'Adam':
        optimizer = torch.optim.Adam(enet.parameters(), lr=lr, weight_decay=wd)
    elif optim == 'SGD':
        optimizer = torch.optim.SGD(enet.parameters(), lr=lr, weight_decay=wd)
    print ('[INFO]Defined the loss function and the optimizer')

    # Training Loop starts
    print ('[INFO]Staring Training...')
    
    # Assuming we are using the CamVid Dataset
    bc_train = len(os.listdir(ip)) // batch_size
    bc_eval = len(os.listdir(ipv)) // batch_size

    pipe = loader(ip, lp, batch_size, rh, rw)
    eval_pipe = loader(ipv, lpv, batch_size, rh, rw)

    loss_list = []
    iou_list = []
    for e in range(1, epochs+1):
        print("\n")
        print ('-'*15,'Epoch %d' % e, '-'*15)
        enet.train()
        
        for _ in tqdm(range(bc_train)):
            X_batch, mask_batch = next(pipe)            
            X_batch, mask_batch = X_batch.to(device), mask_batch.to(device)

            optimizer.zero_grad()

            out = enet(X_batch.float())

            loss = criterion(out, mask_batch.long())
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

        if (e+1) % print_every == 0:
            print ('Epoch {}/{}...'.format(e, epochs), 'Loss {:6f}'.format(train_loss))
        
        if e % eval_every == 0:
            eval_losses = []
            with torch.no_grad():
                enet.eval()
                conf_mat = np.zeros((num_classes, num_classes), dtype=np.float64)
                for _ in tqdm(range(bc_eval)):
                    inputs, labels = next(eval_pipe)
                    inputs, labels = inputs.to(device), labels.to(device)
                    out = enet(inputs)
                    _, preds = torch.max(out, 1)

                    conf_mat += confusion_matrix(labels.cpu().int().numpy(), preds.cpu().float().detach().int().numpy(), num_classes)
                    loss = criterion(out, labels.long())

                    eval_loss = loss.item()
                    eval_losses.append(eval_loss)
            avg_val_loss = np.mean(eval_losses)
            conf_matrix_dict, avg_precision, avg_recall, avg_iou = evaluate_confusion_matrix(conf_mat, num_classes)

        if e == 1:
            best_loss = train_loss
            best_iou = avg_iou
            loss_list.append(train_loss)
            iou_list.append(avg_iou)
            checkpoint = {'state_dict' : enet.state_dict()}
            torch.save(checkpoint, '{}/best_model.pth'.format(save_path))
        elif e != 1:
            if avg_iou >= best_iou: #train_loss <= best_loss
                # print("Better Loss Found! Model saved!")
                # best_loss = train_loss
                # loss_list.append(train_loss)
                print("Better IoU Found! Model saved!")
                best_iou = avg_iou
                iou_list.append(avg_iou)
                with torch.no_grad():
                    enet.eval()
                    conf_mat = np.zeros((num_classes, num_classes), dtype=np.float64)
                    for _ in tqdm(range(bc_eval)):
                        inputs, labels = next(eval_pipe)
                        inputs, labels = inputs.to(device), labels.to(device)
                        out = enet(inputs)
                        _, preds = torch.max(out, 1)

                        conf_mat += confusion_matrix(labels.cpu().int().numpy(), preds.cpu().float().detach().int().numpy(), num_classes)
                        loss = criterion(out, labels.long())
                        eval_loss = loss.item()
                conf_matrix_dict, avg_precision, avg_recall, avg_iou = evaluate_confusion_matrix(conf_mat, num_classes)
                print("Average Precision: {}, Average Recall: {}, Average IoU: {}".format(avg_precision, avg_recall, avg_iou))

                checkpoint = {'state_dict' : enet.state_dict()}
                torch.save(checkpoint, '{}/best_model.pth'.format(save_path))

        if e % save_every == 0:
            checkpoint = {'epochs' : e, 'state_dict' : enet.state_dict()}
            torch.save(checkpoint, '{}/ckpt-enet-{}.pth'.format(save_path, e))
            print('Model saved!')

    print ('[INFO]Training Process complete!')

# python init.py --mode train -iptr ./content/camvid/train/ -lptr ./content/camvid/trainannot/