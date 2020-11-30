#vaemodel
import copy
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils import data
from data_loader import DATA_LOADER as dataloader
import final_classifier as  classifier
import models

class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim,nclass)
        self.logic = nn.LogSoftmax(dim=1)
        self.lossfunction =  nn.NLLLoss()

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o

class Model(nn.Module):

    def __init__(self,hyperparameters):
        super(Model,self).__init__()
        self.save_dir=hyperparameters['save_dir']
        self.device = hyperparameters['device']
        self.auxiliary_data_source = hyperparameters['auxiliary_data_source']
        self.all_data_sources  = ['resnet_features',self.auxiliary_data_source]
        self.DATASET = hyperparameters['dataset']
        self.num_shots = hyperparameters['num_shots']
        self.latent_size = hyperparameters['latent_size']
        self.batch_size = hyperparameters['batch_size']
        self.hidden_size_rule = hyperparameters['hidden_size_rule']
        self.warmup = hyperparameters['model_specifics']['warmup']
        self.generalized = hyperparameters['generalized']
        self.classifier_batch_size = 32
        self.img_seen_samples= hyperparameters['samples_per_class'][self.DATASET][0]
        self.att_seen_samples= hyperparameters['samples_per_class'][self.DATASET][1]
        self.att_unseen_samples = hyperparameters['samples_per_class'][self.DATASET][2]
        self.img_unseen_samples = hyperparameters['samples_per_class'][self.DATASET][3]
        self.reco_loss_function = hyperparameters['loss']
        self.nepoch = hyperparameters['epochs']
        self.lr_cls = hyperparameters['lr_cls']
        self.cross_reconstruction = hyperparameters['model_specifics']['cross_reconstruction']
        self.cls_train_epochs = hyperparameters['cls_train_steps']
        self.dataset = dataloader(self.DATASET, copy.deepcopy(self.auxiliary_data_source), device= self.device)
        if hyperparameters['cls_option']==False:
            self.txt = open(self.save_dir + 'result'+'_'+self.DATASET+'.txt', 'w')
        else:
            self.txt = open(self.save_dir + 'result_cls' + '_' + self.DATASET + '.txt', 'w')

        self.txt.writelines(self.DATASET)
        self.txt.writelines(30*'-')
        #self.att_add=models.encoder_att_add(self.dataset.aux_data.size(1),640,self.device)

        if self.DATASET=='CUB':
            self.num_classes=200
            self.num_novel_classes = 50
        elif self.DATASET=='SUN':
            self.num_classes=717
            self.num_novel_classes = 72
        elif self.DATASET=='AWA1' or self.DATASET=='AWA2':
            self.num_classes=50
            self.num_novel_classes = 10
        elif self.DATASET=='APY' :
            self.num_classes=32
            self.num_novel_classes = 12

        feature_dimensions = [2048, self.dataset.aux_data.size(1)]

        self.encoder = {}
        for datatype, dim in zip(self.all_data_sources,feature_dimensions):
            self.encoder[datatype] = models.encoder_template(dim,self.latent_size,self.hidden_size_rule[datatype],self.device)
            print(str(datatype) + ' ' + str(dim))
        #print('latent size',self.latent_size)
        #self.encoder[self.auxiliary_data_source] = models.encoder_att(self.dataset.aux_data.size(1),self.latent_size,self.device)

        self.decoder = {}
        for datatype, dim in zip(self.all_data_sources,feature_dimensions):
            self.decoder[datatype] = models.decoder_template(self.latent_size,dim,self.hidden_size_rule[datatype],self.device)


        parameters_to_optimize = list(self.parameters())
        for datatype in self.all_data_sources:
            parameters_to_optimize +=list(self.encoder[datatype].parameters())
            parameters_to_optimize +=list(self.decoder[datatype].parameters())

        self.optimizer= optim.Adam(parameters_to_optimize ,lr=hyperparameters['lr_gen_model'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

        if self.reco_loss_function=='l2':
            self.reconstruction_criterion = nn.MSELoss(size_average=False)
        elif self.reco_loss_function=='l1':
            self.reconstruction_criterion = nn.L1Loss(size_average=False)

        self.KD=nn.L1Loss(size_average=False)

    def reparameterize(self, mu, logvar):
        if self.reparameterize_with_noise:
            sigma = torch.exp(logvar)
            eps = torch.cuda.FloatTensor(logvar.size()[0],1).normal_(0,1)
            eps  = eps.expand(sigma.size())
            return mu + sigma*eps
        else:
            return mu

    def forward(self):
        pass

    def map_label(self,label, classes):
        mapped_label = torch.LongTensor(label.size()).to(self.device)
        for i in range(classes.size(0)):
            mapped_label[label==classes[i]] = i

        return mapped_label

    def trainstep(self, img, att, pro_encoder):
        mu_img, logvar_img = self.encoder['resnet_features'](img)

        z_from_img = self.reparameterize(mu_img, logvar_img)

        mu_att, logvar_att = self.encoder[self.auxiliary_data_source](att)
        z_from_att = self.reparameterize(mu_att, logvar_att)

        img_from_img = self.decoder['resnet_features'](z_from_img)
        att_from_att = self.decoder[self.auxiliary_data_source](z_from_att)

        reconstruction_loss = self.reconstruction_criterion(img_from_img, img) + self.reconstruction_criterion(att_from_att, att)

        img_from_att = self.decoder['resnet_features'](z_from_att)
        att_from_img = self.decoder[self.auxiliary_data_source](z_from_img)

        cross_reconstruction_loss = self.reconstruction_criterion(img_from_att, img) + self.reconstruction_criterion(att_from_img, att)

        KLD = (0.5 * torch.sum(1 + logvar_att - mu_att.pow(2) - logvar_att.exp())) + (0.5 * torch.sum(1 + logvar_img - mu_img.pow(2) - logvar_img.exp()))

        distance = torch.sqrt(torch.sum((mu_img - mu_att) ** 2, dim=1) + torch.sum((torch.sqrt(logvar_img.exp()) - torch.sqrt(logvar_att.exp())) ** 2, dim=1))
        distance = distance.sum()

        f1 = 1.0*(self.current_epoch - self.warmup['cross_reconstruction']['start_epoch'] )/(1.0*( self.warmup['cross_reconstruction']['end_epoch']- self.warmup['cross_reconstruction']['start_epoch']))
        f1 = f1*(1.0*self.warmup['cross_reconstruction']['factor'])
        cross_reconstruction_factor = torch.cuda.FloatTensor([min(max(f1,0),self.warmup['cross_reconstruction']['factor'])])
        f2 = 1.0 * (self.current_epoch - self.warmup['beta']['start_epoch']) / ( 1.0 * (self.warmup['beta']['end_epoch'] - self.warmup['beta']['start_epoch']))
        f2 = f2 * (1.0 * self.warmup['beta']['factor'])
        beta = torch.cuda.FloatTensor([min(max(f2, 0), self.warmup['beta']['factor'])])
        f3 = 1.0*(self.current_epoch - self.warmup['distance']['start_epoch'] )/(1.0*( self.warmup['distance']['end_epoch']- self.warmup['distance']['start_epoch']))
        f3 = f3*(1.0*self.warmup['distance']['factor'])
        distance_factor = torch.cuda.FloatTensor([min(max(f3,0),self.warmup['distance']['factor'])])

        self.optimizer.zero_grad()
        if pro_encoder!=None:
            mu_img_pro, logvar_img_pro = pro_encoder(img)

            loss_kd=self.KD(mu_img_pro,mu_img)+self.KD(logvar_img_pro,logvar_img)

            loss = reconstruction_loss - beta * KLD+loss_kd
        else:
            loss = reconstruction_loss - beta * KLD

        if cross_reconstruction_loss>0:
            loss += cross_reconstruction_factor*cross_reconstruction_loss
        if distance_factor >0:
            loss += distance_factor*distance
        loss.backward()

        self.optimizer.step()

        return loss.item()

    def train_vae(self,pro_encoder=None):

        losses = []

        self.dataloader = data.DataLoader(self.dataset,batch_size=self.batch_size,shuffle=True,drop_last=True)

        self.dataset.novelclasses =self.dataset.novelclasses.long().cuda()
        self.dataset.seenclasses =self.dataset.seenclasses.long().cuda()
        self.train()
        self.reparameterize_with_noise = True

        print('train for reconstruction')
        for epoch in range(0, self.nepoch ):
            self.current_epoch = epoch
            i=-1
            for iters in range(0, self.dataset.ntrain, self.batch_size):
                i+=1

                label, data_from_modalities = self.dataset.next_batch(self.batch_size)

                for j in range(len(data_from_modalities)):
                    data_from_modalities[j] = data_from_modalities[j].to(self.device)
                    data_from_modalities[j].requires_grad = False

                loss = self.trainstep(data_from_modalities[0], data_from_modalities[1],pro_encoder)

                if i%50==0:

                    print('epoch ' + str(epoch) + ' | iter ' + str(i) + '\t'+
                    ' | loss ' +  str(loss)[:5]   )

                if i%50==0 and i>0:
                    losses.append(loss)

        # turn into evaluation mode:
        for key, value in self.encoder.items():
            self.encoder[key].eval()
        for key, value in self.decoder.items():
            self.decoder[key].eval()

        return losses



    def train_classifier_final(self, show_plots=False):

        history = []


        cls_seenclasses = self.dataset.seenclasses
        cls_novelclasses = self.dataset.novelclasses


        novelclass_aux_data = self.dataset.novelclass_aux_data
        seenclass_aux_data = self.dataset.seenclass_aux_data

        novel_corresponding_labels = self.dataset.novelclasses.long().to(self.device)
        seen_corresponding_labels = self.dataset.seenclasses.long().to(self.device)

        novel_test_feat = self.dataset.data['test_unseen']['resnet_features']
        seen_test_feat = self.dataset.data['test_seen']['resnet_features']
        test_seen_label = self.dataset.data['test_seen']['labels']
        test_novel_label = self.dataset.data['test_unseen']['labels']


        print('mode: gzsl')
        clf = LINEAR_LOGSOFTMAX(self.latent_size, self.num_classes)
        clf.apply(models.weights_init)

        with torch.no_grad():
            self.reparameterize_with_noise = False

            mu1, var1 = self.encoder['resnet_features'](novel_test_feat)
            test_novel_X = self.reparameterize(mu1, var1).to(self.device).data
            test_novel_Y = test_novel_label.to(self.device)

            mu2, var2 = self.encoder['resnet_features'](seen_test_feat)
            test_seen_X = self.reparameterize(mu2, var2).to(self.device).data
            test_seen_Y = test_seen_label.to(self.device)

            self.reparameterize_with_noise = True

            def sample_train_data_on_sample_per_class_basis(features, label, sample_per_class):
                sample_per_class = int(sample_per_class)

                if sample_per_class != 0 and len(label) != 0:

                    classes = label.unique()
                    for i, s in enumerate(classes):

                        features_of_that_class = features[label == s, :]
                        multiplier = torch.ceil(torch.cuda.FloatTensor([max(1, sample_per_class / features_of_that_class.size(0))])).long().item()
                        features_of_that_class = features_of_that_class.repeat(multiplier, 1)

                        if i == 0:
                            features_to_return = features_of_that_class[:sample_per_class, :]
                            labels_to_return = s.repeat(sample_per_class)
                        else:
                            features_to_return = torch.cat((features_to_return, features_of_that_class[:sample_per_class, :]), dim=0)
                            labels_to_return = torch.cat((labels_to_return, s.repeat(sample_per_class)),dim=0)

                    return features_to_return, labels_to_return
                else:
                    return torch.cuda.FloatTensor([]), torch.cuda.LongTensor([])

            att_unseen_feat, att_unseen_label = sample_train_data_on_sample_per_class_basis(novelclass_aux_data,novel_corresponding_labels,self.att_unseen_samples)
            att_seen_feat, att_seen_label = sample_train_data_on_sample_per_class_basis(seenclass_aux_data,seen_corresponding_labels, self.att_seen_samples)

            def convert_datapoints_to_z(features, encoder):
                if features.size(0) != 0:
                    mu_, logvar_ = encoder(features)
                    z = self.reparameterize(mu_, logvar_)
                    return z
                else:
                    return torch.cuda.FloatTensor([])


            z_seen_att = convert_datapoints_to_z(att_seen_feat, self.encoder[self.auxiliary_data_source])
            z_unseen_att = convert_datapoints_to_z(att_unseen_feat,self.encoder[self.auxiliary_data_source])

            train_Z = [z_seen_att,z_unseen_att]
            train_L = [att_seen_label,att_unseen_label]

            # empty tensors are sorted out
            train_X = [train_Z[i] for i in range(len(train_Z)) if train_Z[i].size(0) != 0]
            train_Y = [train_L[i] for i in range(len(train_L)) if train_Z[i].size(0) != 0]

            train_X = torch.cat(train_X, dim=0)
            train_Y = torch.cat(train_Y, dim=0)

        cls = classifier.CLASSIFIER(clf, train_X, train_Y, test_seen_X, test_seen_Y, test_novel_X,
                                    test_novel_Y,
                                    cls_seenclasses, cls_novelclasses,
                                    self.num_classes, self.device, self.lr_cls, 0.5, 1,
                                    self.classifier_batch_size,
                                    self.generalized)

        for k in range(self.cls_train_epochs):
            if k > 0:
                cls.acc_seen, cls.acc_novel, cls.H = cls.fit()
            print('[%.1f]     novel=%.4f, seen=%.4f, h=%.4f , loss=%.4f' % (k, cls.acc_novel, cls.acc_seen, cls.H, cls.average_loss))
            txt_input='[%.1f]     novel=%.4f, seen=%.4f, h=%.4f , loss=%.4f \n' % (k, cls.acc_novel, cls.acc_seen, cls.H, cls.average_loss)
            self.txt.writelines(txt_input)
            history.append([torch.tensor(cls.acc_seen).item(), torch.tensor(cls.acc_novel).item(),torch.tensor(cls.H).item()])

        return torch.tensor(cls.acc_seen).item(), torch.tensor(cls.acc_novel).item(), torch.tensor(cls.H).item(), history