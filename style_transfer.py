# -*- coding: utf-8 -*-



import torch,torchvision
import numpy as np 
from matplotlib import pyplot as plt
import os
from PIL import Image
from skimage import io
import copy
import tqdm
tensor_to_numpy = lambda t:t.detach().cpu().numpy() 
device = 'cpu'

"""Define functions for converting the input to the CNN back into a visualizable image"""
mean,std = (0.485, 0.456, 0.406),(0.229, 0.224, 0.225)
def denormalize_tensor(t,
                       mean = mean,
                       std = std):
    
    mean = torch.tensor(mean).float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor(std).float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    
    mean = mean.type_as(t).to(t.device)
    std = std.type_as(t).to(t.device)
    
    dnrm_t = t * std + mean
    return dnrm_t
    
def tensor_to_im(t,
                 mean = mean,
                 std=std):
    t = denormalize_tensor(t,
                           mean=mean,
                           std=std)
    t_np = tensor_to_numpy(t)
    im = np.transpose(t_np,(0,2,3,1))
    return im

""" for Gram Matrix"""

def gram_matrix(t):
    b,c,h,w = t.shape
    f = t.view(b,c,-1)
    g = torch.einsum('bxf,bzf->bxz',[f,f])
    g = g*(1./np.prod(t.shape)) # to account for difference in number of channels and pixels at every layer of the CNN
    return g

""" Forward Hook """
def fwdHook(self,input,output):
    self.feat = output
    pass
    
def style_transfer(model,
                  transforms,
                  content,
                   style,
                   nepochs  = 2400,
                   style_lambda = 1000000
                  ):

    if model_name == 'vgg19':
        layers_for_style = [0,
                            2,
                            5,
                            7,
                            10]
        layers_for_content = [0]
    layers_of_interest = layers_for_style[:]
    layers_of_interest.extend(layers_for_content[:])

    hooked_layers = []
    model_children = list(model.children())
    for li in layers_of_interest:
        l = model_children[li]
        hooked_layers.append(l.register_forward_hook(fwdHook))
        
    """Declare the style and content images,  the optimizee and the optimizer"""

    
    output = content.detach().clone() # the output starts from the content image
    content,style,output = content.to(device),style.to(device),output.to(device) # push everything onto the device
    output = output.requires_grad_(True) # requires_grad should be the last operation on the optimizee 
    # opt = torch.optim.LBFGS([output]) # similar to the tutorial, using a L-BFGS optimizer
    opt = torch.optim.Adam([output],
                          lr = 1e-2)
    
    """Optimization loop and visualization"""


    trends = {'total_loss':[],
             'content_loss':[],
             'style_loss':[],}
     
    for e in tqdm.tqdm(range(nepochs)):

        '''----- style -----'''
        _ = model(style) # pass the style image through the CNN
        style_feat = [model_children[li].feat for li in layers_for_style] # get the feats from the style layers    
        style_gram = [gram_matrix(f) for f in style_feat] # calculate the gram matrix over these layers

        '''----- content -----'''
        _ = model(content) # pass the content image through the CNN

        content_feat = model_children[layers_for_content[0]].feat  # there is only one content layer

        '''----- output -----'''
        _ = model(output) # pass the optimizee through the CNN
        output_style_feat = [model_children[li].feat for li in layers_for_style]     # get the optimizee style feats
        output_style_gram = [gram_matrix(f) for f in output_style_feat] # get the optimizee gram matrices
    #     output_content_feat = [model_children[li].feat for li in layers_for_content]
        output_content_feat = model_children[layers_for_content[0]].feat # get the optimizee content features

        '''----- losses -----'''
        content_loss = torch.nn.functional.mse_loss(content_feat.view(-1),output_content_feat.view(-1))  
        style_loss = sum([torch.nn.functional.mse_loss(st_gr.view(-1),op_gr.view(-1))  for st_gr,op_gr in zip(style_gram,output_style_gram)])
        total_loss = content_loss + style_lambda * style_loss

        '''----- update -----'''
        opt.zero_grad()
        total_loss.backward()
        opt.step()

        '''----- bookkeeping -----'''
        trends['total_loss'].append(tensor_to_numpy(total_loss))
        trends['content_loss'].append(tensor_to_numpy(content_loss))
        trends['style_loss'].append(tensor_to_numpy(style_loss))



        pass

    """Save the final result to disk"""
    result = tensor_to_im(output)[0]
    if 'save_to_disk':   
        io.imsave('style_transfered.png',result)
    return result,trends
        

if __name__ == '__main__':
    
    model_specs = {'alexnet':(227,227),
                  'vgg19':(224,224),
                  }
    model_name = 'vgg19'
    
    transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(model_specs[model_name]),
                                                torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                                  std=(0.229, 0.224, 0.225)),
                                                ])
    """Model specifications and transform:
    sizes of images, mean and variance
    """


    """See the images"""

    im_scream,im_bridge_girl = io.imread('scream.jpeg'), io.imread('girl_on_bridge.jpg')

    plt.figure()
    plt.imshow(im_scream)
    plt.title('Scream')

    plt.figure()
    plt.imshow(im_bridge_girl)
    plt.title('Girl On Bridge')

    """make tensors from images, resizing and transforming them."""

    im_scream_pil,im_bridge_girl_pil = Image.fromarray(im_scream),Image.fromarray(im_bridge_girl)

    scream = transforms(im_scream_pil).unsqueeze(0)
    bridge_girl = transforms(im_bridge_girl_pil).unsqueeze(0)
    content,style = bridge_girl,scream
    """Get the model"""

    model = torchvision.models.vgg19(pretrained=True).features.to(device).eval() # get the model in eval mode
    results,trends = style_transfer(model,
                  transforms,
                  content,
                   style,
                   nepochs  = 2400,
                   style_lambda = 1000000
                  )














#"""Some setup code in case we want to make a time lapse of style transfer"""




