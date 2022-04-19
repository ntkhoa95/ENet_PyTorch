import os, cv2, torch
import numpy as np
import matplotlib.pyplot as plt


def loader(img_path, gt_path, batch_size, h=512, w=512):
    """
    Loading inputs and labels
    Args:
        img_path: rgb input image
        gt_path: ground truth label map
        batch_size: number of inputs in a batch
    """
    input_images = os.listdir(img_path)
    gt_images = os.listdir(gt_path)

    assert len(input_images) == len(gt_images)

    if str(batch_size).lower() == 'all':
        batch_size = len(input_images)

    while True:
        batch_indices = np.random.randint(0, len(input_images), batch_size)

        inputs, labels = [], []
        for i in batch_indices:
            img = cv2.imread(os.path.join(img_path, input_images[i]))#.astype('float')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (h, w), cv2.INTER_NEAREST)

            gt = cv2.imread(os.path.join(gt_path, gt_images[i]), -1)#.astype('float')
            gt = cv2.resize(gt, (h, w), cv2.INTER_NEAREST)

            inputs.append(img)
            labels.append(gt)

        inputs = np.stack(inputs, axis=2)
        inputs = torch.FloatTensor(inputs).transpose(0, 2).transpose(1, 3)
        labels = torch.FloatTensor(np.array(labels))

        yield inputs, labels

def decode_camvid_segmap(image):
    sky = [128, 128, 128]
    building = [128, 0, 0]
    pole = [192, 192, 128]
    road_marking = [255, 69, 0]
    road = [128, 64, 128]
    pavement = [60, 40, 222]
    tree = [128, 128, 0]
    sign_symbol = [192, 128, 128]
    fence = [64, 64, 128]
    car = [64, 0, 128]
    pedestrian = [64, 64, 0]
    bicyclist = [0, 128, 192]

    label_colors = np.array([sky, building, pole, road_marking, road, 
                              pavement, tree, sign_symbol, fence, car, 
                              pedestrian, bicyclist]).astype(np.uint8)
    
    r = np.zeros(image.shape, dtype=np.uint8)
    g = np.zeros(image.shape, dtype=np.uint8)
    b = np.zeros(image.shape, dtype=np.uint8)

    for label in range(len(label_colors)):
        r[image == label] = label_colors[label, 0]
        g[image == label] = label_colors[label, 1]
        b[image == label] = label_colors[label, 2]
    
    rgb = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b

    return rgb

def decode_custom_segmap(image):
    class_1 = [0, 0, 255]
    class_2 = [0, 255, 0]
    class_3 = [255, 0, 0]

    label_colors = np.array([class_1, class_2, class_3]).astype(np.uint8)
    
    r = np.zeros(image.shape, dtype=np.uint8)
    g = np.zeros(image.shape, dtype=np.uint8)
    b = np.zeros(image.shape, dtype=np.uint8)

    for label in range(len(label_colors)):
        r[image == label] = label_colors[label, 0]
        g[image == label] = label_colors[label, 1]
        b[image == label] = label_colors[label, 2]
    
    rgb = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b

    return rgb

def show_images(images, in_row=True):
    '''
    Helper function to show 3 images
    '''
    total_images = len(images)

    rc_tuple = (1, total_images)
    if not in_row:
        rc_tuple = (total_images, 1)
    
    plt.figure(figsize=(12, 8))
    for ii in range(len(images)):
        plt.subplot(*rc_tuple, ii+1)
        plt.title(images[ii][0])
        plt.axis('off')
        plt.imshow(images[ii][1])
    plt.show()

def get_class_weights(loader, num_classes, c=1.02):
    '''
    This class return the class weights for each class
    
    Arguments:
    - loader : The generator object which return all the labels at one iteration
               Do Note: That this class expects all the labels to be returned in
               one iteration
    - num_classes : The number of classes
    Return:
    - class_weights : An array equal in length to the number of classes
                      containing the class weights for each class
    '''

    _, labels = next(loader)
    all_labels = labels.flatten()
    each_class = np.bincount(all_labels, minlength=num_classes)
    prospensity_score = each_class / len(all_labels)
    class_weights = 1 / (np.log(c + prospensity_score))
    return class_weights

def confusion_matrix(x , y, n, ignore_label=None, mask=None):
        if mask is None:
            mask = np.ones_like(x) == 1
        k = (x >= 0) & (y < n) & (x != ignore_label) & (mask.astype(np.bool))
        return np.bincount(n * x[k].astype(int) + y[k], minlength=n**2).reshape(n, n)

def get_scores(conf_matrix):
        if conf_matrix.sum() == 0:
            return 0, 0, 0
        with np.errstate(divide='ignore',invalid='ignore'):
            overall = np.diag(conf_matrix).sum() / np.float(conf_matrix.sum())
            perclass = np.diag(conf_matrix) / conf_matrix.sum(1).astype(np.float)
            IU = np.diag(conf_matrix) / (conf_matrix.sum(1) + conf_matrix.sum(0) - np.diag(conf_matrix)).astype(np.float)
        return overall * 100., np.nanmean(perclass) * 100., np.nanmean(IU) * 100.

def evaluate_confusion_matrix(conf_matrix, num_classes):
    labels = list(np.arange(num_classes))
    # Sets the initial for final precision, final recall, final iou
    all_precision, all_recall, all_iou = 0, 0, 0

    # Creates dictionary to save data
    conf_matrix_dict = dict()

    # Loop through each label
    for label in labels:
        # Gets pairs of coordiantes of TRUE NEGATIVE, FALSE POSITIVE, FALSE NEGATIVE
        tn_list = labels.copy()
        del tn_list[label]
        tn_pair = [(i, j) for i in tn_list for j in tn_list]
        fp_pair = [(label, j) for j in tn_list]
        fn_pair = [(j, label) for j in tn_list]
        
        # Gets the TRUE POSITIVE value
        TP = conf_matrix[label, label]
        # Gets the TRUE NEGATIVE value
        TN = 0
        for pair in tn_pair:
            value = conf_matrix[pair]
            TN += value
        # Gets the FALSE POSITIVE value
        FP = 0
        for pair in fp_pair:
            value = conf_matrix[pair]
            FP += value
        # Gets the FALSE NEGATIVE value
        FN = 0
        for pair in fn_pair:
            value = conf_matrix[pair]
            FN += value

        # Calculates the Precision
        precision = TP / (TP + FP) * 100

        # Calculates the recall
        recall = TP / (TP + FN) * 100

        # Calculates the IoU
        IoU = TP / (TP +FP + FN) * 100

        all_precision += precision
        all_recall += recall
        all_iou += IoU

        conf_matrix_dict[f"Label_{label}"] = {"Precision": round(precision, 2), "Recall": round(recall, 2), "IoU": round(IoU, 2)}
    
    return conf_matrix_dict, round(all_precision/num_classes, 2), round(all_recall/num_classes, 2), round(all_iou/num_classes, 2)