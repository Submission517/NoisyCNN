import matplotlib.pyplot as plt
import os
import numpy as np
import copy


#noise strength
def LinearNoiseStrength(dataset):
        storeDict = {'resnet18':{'pratios':[], 'vaccs':[]}, 'resnet34':{'pratios':[], 'vaccs':[]}, 'resnet50':{'pratios':[], 'vaccs':[]}, 'resnet101':{'pratios':[], 'vaccs':[]} }
        modelsdir = './saved_models'
        models = os.listdir(modelsdir)
        pratios = []
        vaccs = []
        for key in storeDict.keys():
                for model in models:
                        saved_dataset = model.split('_')[1]
                        if(dataset == saved_dataset and key in model and 'impulse' not in model and 'gaussian' not in model):
                                vanilla = (model.split('_')[5]).split('.')[0]
                                mscale = model.split('_')[2]
                                vaccs = float(model.split('_')[0])
                                if(vanilla == 'vanilla'):
                                        pratio = 0.0
                                        if(pratio in storeDict[mscale]['pratios']):
                                                continue
                                        storeDict[mscale]['pratios'].append(pratio)
                                        storeDict[mscale]['vaccs'].append(vaccs)
                                else:
                                        layer = int(model.split('_')[10])
                                        if( layer != 4):
                                                continue
                                        pratio = float(model.split('_')[8])
                                        if(pratio in storeDict[mscale]['pratios']):
                                                continue
                                        storeDict[mscale]['pratios'].append(pratio)
                                        storeDict[mscale]['vaccs'].append(vaccs)

        for key in storeDict.keys():
                reorg_index = np.argsort(storeDict[key]['pratios'])
                pratios = sorted(storeDict[key]['pratios'])
                reorg_vaccs = copy.deepcopy(storeDict[key]['vaccs'])
                i = 0
                for index in reorg_index:
                        reorg_vaccs[i] = storeDict[key]['vaccs'][index]
                        i += 1
                storeDict[key]['vaccs'] = reorg_vaccs
                storeDict[key]['pratios'] = pratios

        
        font = {'family':'serif',
                #'style':'italic',
                'weight':'normal',
                'color':'black',
                'size':70}
        for key in storeDict.keys():
                if(key == 'resnet18'):
                        plt.plot(storeDict[key]['pratios'], storeDict[key]['vaccs'], label = key, marker = 'o', c = 'blue', markersize = 40) #marker = 'o', c = 'blue',
                elif(key == 'resnet34'):
                        plt.plot(storeDict[key]['pratios'], storeDict[key]['vaccs'], label = key, marker = 'o', c = 'red', markersize = 40) #marker = 'o', c = 'blue',
                elif(key == 'resnet50'):
                        plt.plot(storeDict[key]['pratios'], storeDict[key]['vaccs'], label = key, marker = 'o', c = 'cyan', markersize = 40) #marker = 'o', c = 'blue',
                elif(key == 'resnet101'):
                        plt.plot(storeDict[key]['pratios'], storeDict[key]['vaccs'], label = key, marker = 'o', c = 'magenta', markersize = 40) #marker = 'o', c = 'blue',
        plt.xlim(0, 0.9)
        plt.xticks(size= 70)
        plt.yticks(size= 70)
        plt.xlabel('Linear Noise Strength', fontdict=font)
        plt.ylabel('Accuracy', fontdict=font)
        plt.legend(prop={'size':70})
        plt.show()


#Inject Layer
def LinearNoiseLayer(dataset='ImageNet', stren = 0.4):
        storeDict = {'resnet18':{'layer':[], 'vaccs':[]}, 'resnet34':{'layer':[], 'vaccs':[]}, 'resnet50':{'layer':[], 'vaccs':[]}, 'resnet101':{'layer':[], 'vaccs':[]} }
        modelsdir = './saved_models'
        models = os.listdir(modelsdir)
        pratios = []
        vaccs = []
        for key in storeDict.keys():
                for model in models:
                        saved_dataset = model.split('_')[1]
                        if(dataset == saved_dataset and key in model and 'impulse' not in model and 'gaussian' not in model):
                                vanilla = (model.split('_')[5]).split('.')[0]
                                mscale = model.split('_')[2]
                                vaccs = float(model.split('_')[0])
                                if(vanilla == 'vanilla'):
                                        pratio = 0.0
                                        if(pratio in storeDict[mscale]['layer']):
                                                continue
                                        storeDict[mscale]['layer'].append(pratio)
                                        storeDict[mscale]['vaccs'].append(vaccs)
                                else:
                                        strenth = float(model.split('_')[8])
                                        layer = float(model.split('_')[10])
                                        for temp_model in models:
                                                if('vanilla' not in temp_model):
                                                        temp_layer = float(temp_model.split('_')[10])
                                                        temp_vaccs = float(temp_model.split('_')[0])
                                                        temp_mscale = temp_model.split('_')[2]
                                                        if(temp_vaccs > vaccs and temp_layer == layer and temp_mscale == mscale):
                                                                vaccs = temp_vaccs
                                                #if( abs(strenth - stren)<0.2):
                                        if(layer in storeDict[mscale]['layer']):
                                                continue
                                        storeDict[mscale]['layer'].append(layer)
                                        storeDict[mscale]['vaccs'].append(vaccs)

        for key in storeDict.keys():
                reorg_index = np.argsort(storeDict[key]['layer'])
                layers = sorted(storeDict[key]['layer'])
                reorg_vaccs = copy.deepcopy(storeDict[key]['vaccs'])
                i = 0
                for index in reorg_index:
                        reorg_vaccs[i] = storeDict[key]['vaccs'][index]
                        i += 1
                storeDict[key]['vaccs'] = reorg_vaccs
                storeDict[key]['layer'] = layers

        
        font = {'family':'serif',
                #'style':'italic',
                'weight':'normal',
                'color':'black',
                'size':70}
        for key in storeDict.keys():
                if(key == 'resnet18'):
                        plt.plot(storeDict[key]['layer'], storeDict[key]['vaccs'], label = key, marker = 'o', c = 'blue', markersize = 40) #marker = 'o', c = 'blue',
                elif(key == 'resnet34'):
                        plt.plot(storeDict[key]['layer'], storeDict[key]['vaccs'], label = key, marker = 'o', c = 'red', markersize = 40) #marker = 'o', c = 'blue',
                elif(key == 'resnet50'):
                        plt.plot(storeDict[key]['layer'], storeDict[key]['vaccs'], label = key, marker = 'o', c = 'cyan', markersize = 40) #marker = 'o', c = 'blue',
                elif(key == 'resnet101'):
                        plt.plot(storeDict[key]['layer'], storeDict[key]['vaccs'], label = key, marker = 'o', c = 'magenta', markersize = 40) #marker = 'o', c = 'blue',
        plt.xlim(0, 4)
        plt.xticks(size= 70)
        plt.yticks(size= 70)
        plt.xlabel('Injected Layer', fontdict=font)
        plt.ylabel('Accuracy', fontdict=font)
        plt.legend(prop={'size':70})
        plt.show()
        

def remove_duplicated_saved_models():
        modelsdir = './saved_models'
        models = os.listdir(modelsdir)
        del_models = []
        for model in models:
                acc = float(model.split('_')[0])
                model_id = model.split('_')[1:]
                if(model in del_models):
                        continue
                for temp_model in models:
                        temp_acc = float(temp_model.split('_')[0])
                        temp_model_id = temp_model.split('_')[1:]
                        if(temp_acc < acc and temp_model_id == model_id):
                                del_models.append(temp_model)
                                os.remove(modelsdir+'/'+temp_model)
                                print('del', temp_model)


if __name__ == '__main__':
        print('halo')
        remove_duplicated_saved_models()
        LinearNoiseStrength('ImageNet')
        LinearNoiseLayer('ImageNet')
