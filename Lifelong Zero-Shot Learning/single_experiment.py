import copy
from vaemodel import Model
import torch
import os
import shutil
torch.cuda.set_device(3)

mode='DEN+KD'
ZERO_THRESHOLD = 1e-4
L2_COEFF = 5e-3
ALL_CLASSES=64

num_seen=200
num_unseen=400
hyperparameters = {
'save_dir': '/home/weikun/code/ijcai_new/7.14'+'/'+mode+'/',
'cls_option':  False,
'num_shots': 0,
'device': 'cuda',
'model_specifics': {'cross_reconstruction': True,
                    'name': 'CADA',
                    'distance': 'wasserstein',
                  'warmup': {'beta': {'factor': 0.25,
                                        'end_epoch': 93,
                                        'start_epoch': 0},
                               'cross_reconstruction': {'factor': 2.37,
                                                        'end_epoch': 75,
                                                        'start_epoch': 21},
                               'distance': {'factor': 8.13,
                                            'end_epoch': 22,
                                            'start_epoch': 6}}},
'lr_gen_model': 0.00015,
'generalized': True,
'batch_size': 50,
'epochs': 100,
# 'epochs': 100,
'loss': 'l1',
'auxiliary_data_source': 'attributes',
'lr_cls': 0.001,
'dataset': 'CUB',
'hidden_size_rule': {'resnet_features': (1560, 1660),
                     'attributes': (1450, 665),
                     'sentences': (1450, 665)},
'latent_size': 64
}

cls_train_steps = [
{'dataset': 'SUN', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 50},
{'dataset': 'AWA1', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 70},
{'dataset': 'CUB', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 70},
{'dataset': 'APY', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 30},
]

class my_hook(object):
    def __init__(self, mask1):
        self.mask1 = torch.Tensor(mask1).long().nonzero().view(-1).numpy()
    def __call__(self, grad):
        grad_clone = grad.clone()
        if self.mask1.size:
            grad_clone[:, self.mask1] = 0
        return grad_clone


def select_neurons(model):
    layers = []
    for name, param in model.named_parameters():
        if 'bias' not in name:
            layers.append(param)
    layers = reversed(layers)
    hooks = []
    selected=[]
    for layer in layers:
        x_size, y_size = layer.size()
        active = [True] * y_size
        data = layer.data
        for x in range(x_size):
            for y in range(y_size):
                weight = data[x, y]
                if (weight > ZERO_THRESHOLD):
                    active[y] = False
        h = layer.register_hook(my_hook(active))
        hooks.append(h)
        selected.append((y_size - sum(active), y_size))
    for nr, (sel, neurons) in enumerate(reversed(selected)):
        print("layer %d: %d / %d" % (nr + 1, sel, neurons))
    return hooks

try:
    os.makedirs(hyperparameters['save_dir'])
except:
    pass
shutil.copy('./data_loader.py', hyperparameters['save_dir'])
shutil.copy('./final_classifier.py', hyperparameters['save_dir'])
shutil.copy('./models.py', hyperparameters['save_dir'])
shutil.copy('./single_experiment.py', hyperparameters['save_dir'])
shutil.copy('./vaemodel.py', hyperparameters['save_dir'])
dataset_list=['APY','AWA1', 'CUB', 'SUN']

for dataset_select in dataset_list:
    dataset_select_pro=dataset_list[dataset_list.index(dataset_select)-1]
    print('dataset_select_pro',dataset_select_pro)
    hyperparameters['dataset'] = dataset_select

    hyperparameters['cls_train_steps'] = [x['cls_train_steps'] for x in cls_train_steps
                                          if all([hyperparameters['dataset'] == x['dataset'],
                                                  hyperparameters['generalized'] == x['generalized']])][0]
    hyperparameters['samples_per_class'] = {'CUB': (200, 200, 400, 400), 'SUN': (200, 200, 400, 400),
                            'APY': (200, 200,  400, 400), 'AWA1': (200, 200, 400, 400),
                            'AWA2': (200, 200, 400, 400), 'FLO': (200, 200, 400, 400)}
    if dataset_select == 'APY':
        model = Model(hyperparameters)
        model.to(hyperparameters['device'])
        losses = model.train_vae()


    if dataset_select!='APY':
        model = Model(hyperparameters)
        model.to(hyperparameters['device'])
        model.encoder['resnet_features'].load_state_dict(torch.load(
            hyperparameters['save_dir'] + dataset_select_pro + '_' + 'resnet_features' + '_encoder' + '.t7'))
        model.decoder['resnet_features'].load_state_dict(torch.load(
            hyperparameters['save_dir'] + dataset_select_pro + '_' + 'resnet_features' + '_decoder' + '.t7'))
        model_encoder_copy = copy.deepcopy(model.encoder['resnet_features'])
        #

        hyperparameters_pro = hyperparameters
        hyperparameters_pro['dataset'] = dataset_select_pro
        model_pro = Model(hyperparameters_pro)
        model_pro.encoder['resnet_features'].load_state_dict(torch.load(hyperparameters['save_dir'] + dataset_select_pro + '_' + 'resnet_features' + '_encoder' + '.t7'))
        model_pro.eval()
        model_pro.to(hyperparameters['device'])
        params = list(model.encoder['resnet_features'].parameters())
        for param in params[:-2]:
            param.requires_grad = False
        losses = model.train_vae(model_pro.encoder['resnet_features'])
        for param in model.encoder['resnet_features'].parameters():
            param.requires_grad = True
        hooks_encoder = select_neurons(model.encoder['resnet_features'])
        losses = model.train_vae(model_pro.encoder['resnet_features'])

        for hook in hooks_encoder:
            hook.remove()

    state ={'state_dict': model.state_dict(),
            'hyperparameters':hyperparameters,
            'encoder':{},
            'decoder':{}}
    for d in model.all_data_sources:
        state['encoder'][d] = model.encoder[d].state_dict()
        state['decoder'][d] = model.decoder[d].state_dict()
        torch.save(state['encoder'][d],hyperparameters['save_dir']+dataset_select+'_'+d+'_encoder'+'.t7')
        torch.save(state['decoder'][d],hyperparameters['save_dir']+dataset_select+'_'+d+'_decoder'+'.t7')

cls_train_steps = [
{'dataset': 'SUN', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 50},
{'dataset': 'AWA1', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 70},
{'dataset': 'CUB', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 70},
{'dataset': 'APY', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 30},
]
for dataset_select_cls in dataset_list:
    hyperparameters['dataset']=dataset_select_cls
    hyperparameters['cls_option'] = True
    hyperparameters['cls_train_steps'] = [x['cls_train_steps'] for x in cls_train_steps
                                          if all([hyperparameters['dataset'] == x['dataset'],
                                                  hyperparameters['generalized'] == x['generalized']])][0]
    hyperparameters['samples_per_class'] = {'CUB': (num_seen, num_seen, num_unseen, num_unseen), 'SUN': (num_seen, num_seen, num_unseen, num_unseen),
                            'APY': (num_seen, num_seen, num_unseen, num_unseen), 'AWA1': (num_seen, num_seen, num_unseen, num_unseen),
                            'AWA2': (num_seen, num_seen, num_unseen, num_unseen), 'FLO': (num_seen, num_seen, num_unseen, num_unseen)}
    model = Model(hyperparameters)
    model.encoder['resnet_features'].load_state_dict(torch.load(hyperparameters['save_dir']+ 'SUN' + '_' + 'resnet_features' + '_encoder' + '.t7'))
    model.encoder['attributes'].load_state_dict(torch.load(hyperparameters['save_dir']+ dataset_select_cls + '_' + 'attributes' + '_encoder' + '.t7'))
    model.to(hyperparameters['device'])
    model.eval()
    u,s,h,history = model.train_classifier_final()
    acc = [hi[2] for hi in history]
    print(acc[-1])