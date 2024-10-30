import matplotlib
import pandas as pd
import numpy as np
import sklearn
from conf import path, device, EPOCHS, transform, BATCH_SIZE
from dataSetPreperation import TrainData, TestData, TrainDataLoader, TestDataLoader

from tqdm import tqdm
import torch
import torchvision
from Net import Net

if __name__ == '__main__':
    net = Net()
    print(net.to(device))

    lossF = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    history = {'Test Loss': [], 'Test Accuracy': []}
    for epoch in range(1, EPOCHS + 1):
        # create processBar
        processBar = tqdm(TrainDataLoader, unit='step')
        # set the net mode as train
        net.train(True)

        for step, (trainImgs, labels) in enumerate(processBar):
            # send trainImgs and labels to device
            trainImgs = trainImgs.to(device)
            labels = labels.to(device)
            # make net's grad to 0
            net.zero_grad()
            # forward
            outputs = net(trainImgs)

            # calculate the loss
            loss = lossF(outputs, labels)
            # calculate the prediction
            predictons = torch.argmax(outputs, dim=1)

            accuracy = torch.sum((predictons == labels) / labels.shape[0])

            loss.backward()

            optimizer.step()
            # visualize the training process
            processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" %
                                       (epoch, EPOCHS, loss.item(), accuracy.item()))

            if step == len(processBar) - 1:
                correct, totalLoss = 0, 0
                net.train(False)
                for testImgs, labels in TestDataLoader:
                    testImgs = testImgs.to(device)
                    labels = labels.to(device)
                    outputs = net(testImgs)
                    loss = lossF(outputs, labels)
                    predictons = torch.argmax(outputs, dim=1)

                    totalLoss += loss
                    correct += torch.sum(predictons == labels)
                testAccuracy = correct / (BATCH_SIZE * len(TestDataLoader))
                testLoss = totalLoss / len(TestDataLoader)
                history['Test Loss'].append(testLoss.item())
                history['Test Accuracy'].append(testAccuracy.item())
                processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" %
                                           (epoch, EPOCHS, loss.item, accuracy.item(), testLoss.item(),
                                            testAccuracy.item()))

    # visualize the test loss
    matplotlib.pyplot.plot(history['Test Loss'], label='Test Loss')
    matplotlib.pyplot.legend(loc='best')
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.xlabel('Epoch')
    matplotlib.pyplot.ylabel('Loss')
    matplotlib.pyplot.show()

    # visualize the test accuracy
    matplotlib.pyplot.plot(history['Test Accuracy'], color='red', label='Test Accuracy')
    matplotlib.pyplot.legend(loc='best')
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.xlabel('Epoch')
    matplotlib.pyplot.ylabel('Accuracy')
    matplotlib.pyplot.show()
    torch.save(net, './model.pth')

