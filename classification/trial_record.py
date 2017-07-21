dict_init = {'trial' : 4,
             'file_name':'Latin.txt',
             'path': '',
            'layer_num' : 64,
            'layer_size' : 16,
            'dense_layer_num' : 128,
            'input_shape' : (32, 32, 1),
            'output_num' : 26,
            'ratio_dropout' : [0.3, 0.5],
            'reg' : [0.005, 0.003],
            'opt_name' : 'Adadelta',
            'loss' : 'categorical_crossentropy',
            'metric' : ['accuracy'],
            'activation' : ['relu', 'softmax'],
            'layer_name': ['Conv2D', 'Dense'],
            'epoch' : 30,
            'min_batch' : 50,
            'init_data' : 'Latin alphabet recognition',
             'overall_layers' : 12
        }

def trial_record(hist, dic=dict_init, print_option=True, ttime='Unknown'):
    for key in dict_init.keys():
        if key not in dic.keys():
            dic[key] = dict_init[key]
        exec('%s="%s"'%(key, dic[key]), globals())
    print(trial)
    data = '''
    ###################################
    Trial_{0}
    acc = {1:0.4f}, val_acc = {2:0.4f}
    Training time = {19:0.4f}
    Setting:
    Number of layers = {3} for {14}, {4} for {15}
    Total number of layers = {18}
    Layer size = ({6}, {6})
    ratio of dropout = {5}
    Input shape = {7}
    Output number = {8} 
    Regularization = {9}
    Optimizer = {10}
    Loss function = {11}
    Metric = {12}
    Activation functions = {13}
    Number of epochs = {16}
    Batch size = {17}
    ###################################

            '''.format(trial, hist.history['acc'][-1], hist.history['val_acc'][-1], layer_num, dense_layer_num,
                       ratio_dropout, layer_size, input_shape, output_num, reg, opt_name, loss, metric, activation,
                       layer_name[0], layer_name[1], epoch, min_batch, overall_layers, ttime)
    if print_option: print(data)

    try:
        with open('%s'%file_name, 'r') as f:
            lines = f.readlines()
        if '    Trial_{}\n'.format(trial) in lines:
            lines = lines[:lines.index('Trial_{}'.format(trial))-1]
            dt = ''
            for line in lines:
                dt += '%s\n'%line
            with open('%s'%file_name, 'w') as f:
                f.write(dt +'\n' + data)
        else:
            with open('%s'%file_name, 'a') as f:
                f.write(data)
    except:
        with open('%s'%file_name, 'w') as f:
            f.write(init_data + '\n' + data)

def savefigure(hist, save_plot=True, trial=0, file_name='Latin_', path=''):
    from matplotlib.pyplot import savefig
    import matplotlib.pylab as plt
    plt.figure(figsize=(8, 15))
    plt.subplot(211)
    plt.plot(hist.history['loss'], label='loss')
    plt.title('loss')
    plt.legend()
    plt.subplot(212)
    plt.title('accuracy')
    plt.plot(hist.history["acc"], label="training accuracy")
    plt.plot(hist.history["val_acc"], label='test accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

    if (save_plot) & trial:
        savefig('{2}{0}_{1}.png'.format(file_name, trial, path))
    elif save_plot:
        savefig('{1}{0}.png'.format(file_name, path))

if __name__ == "__main__":
    info = """
    List of functions
    
    - trial_record(hist, dic=dict_init, print_option=True, ttime='Unknown')
        
        Saving record of deep learning results 
        
        hist: keras' deep learning History object which gets returned by the fit method of models.
        dic: Dictionary object which stores keras' deep learning setting. Followings is example and initial value of it.
        dict_init = {'trial' : 4,
                 'file_name':'Latin.txt',
                 'path': '',
                'layer_num' : 64,
                'layer_size' : 16,
                'dense_layer_num' : 128,
                'input_shape' : (32, 32, 1),
                'output_num' : 26,
                'ratio_dropout' : [0.3, 0.5],
                'reg' : [0.005, 0.003],
                'opt_name' : 'Adadelta',
                'loss' : 'categorical_crossentropy',
                'metric' : ['accuracy'],
                'activation' : ['relu', 'softmax'],
                'layer_name': ['Conv2D', 'Dense'],
                'epoch' : 30,
                'min_batch' : 50,
                'init_data' : 'Latin alphabet recognition',
                 'overall_layers' : 12
            }  
        print_option: For the option, True, you can print the record to be saved.
        ttime: Additionally, you can save how long does your code take.
        
        The result has following format.
        
        Latin alphabet recognition
        
        ###################################
        Trial_4
        acc = 0.9846, val_acc = 0.9513
        Training time = 54.66
        Setting:
        Number of layers = 64 for Conv2D, 128 for Dense
        Total number of layers = 12
        Layer size = (16, 16)
        ratio of dropout = [0.3, 0.4, 0.5]
        Input shape = (36, 36, 1)
        Output number = 26
        Regularization = [0.005, 0.003]
        Optimizer = Adadelta
        Loss function = categorical_crossentropy
        Metric = ['accuracy']
        Activation functions = ['relu', 'softmax']
        Number of epochs = 30
        Batch size = 50
        ###################################
        
    - savefigure(hist, save_plot=True, trial=4, file_name='Latin_', path='')
    
        Show graphs of loss function for each epochs and accuracy of train and test data for each epochs 
        
        hist: keras' deep learning History object which gets returned by the fit method of models.
        save_plot: option whether save or not.
        trial: set the number of trial to file name. Optional.
        file_name: set the name of file to save your figure. Recommend to fix the name.
        path: set path. Optional
    """
    print(info)
