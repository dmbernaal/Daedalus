# CREDIT: https://github.com/jav0927/course-v3/blob/master/SSD_Object_Detection_RS50_V2_0_Fixed.ipynb
import numpy as np

def is_unfreeze(learner):
    '''
    Determines whether the next-to-last layer in the model is set to unfreeze or freeze
    '''
    c = 0
    for each in list(learner.model.body[-1][0].parameters()):
        if each.requires_grad: c += 1   
    if c == len(list(learner.model.body[-1][0].parameters())):
        return True 
    else:
        return False

def find_optimal_lr(learner, noise=1, show_df=None, show_min_values=False):
    '''
    Parameters
    
      learner:  The learner (mandatory)
      
      (Optional)
      noise:   Filtering parameter, set to 3. Suggest no to modify this value
      
      show_df: 'head' - Show the top 50 rows, 
               'tail' - Show the tail 50 rows
    
      show_min_values: True  - Display all values, min, and max 
                       False - Display min_loss and max_grad values
    
    Returns:
      optimun_lr - if freeze = True
      Suggested Best Slice - if freeze = False
      
    Author:  J. Adolfo Villalobos @ 2019  
    '''
    
    # Get loss values, corresponding gradients, and lr values from model.recorder
    loss = np.array(learner.recorder.losses)
    loss_grad = np.gradient(loss)   
    # Transform lrs list to np array
    lrs = np.array(learner.recorder.lrs, dtype='float32')
    
    # Create a DataFrame with the data
    data = {'loss': loss.T, 'loss_grad': loss_grad.T, 'lrs': lrs.T}
    df = pd.DataFrame(data, columns=['loss', 'loss_grad', 'lrs', 'min_loss', 'max_loss', 'min_grad', 'max_grad'])
      
    # Populate "min" and "max" columns for loss and gradient values filtering the noise with argrelextrema.     
    from scipy.signal import argrelextrema
    
    #********
    # IMPORTANT: n filters noise (sharp spikes in the data). Higher n value filters noise more aggressively. 
    # n = 3 seems to work best
    n=noise    
    #********
    
    df.min_loss = df.iloc[argrelextrema(df.loss.values, np.less_equal, order=n)[0]]['loss']
    df.max_loss = df.iloc[argrelextrema(df.loss.values, np.greater_equal, order=n)[0]]['loss']
    df.min_grad = df.iloc[argrelextrema(df.loss_grad.values, np.less_equal, order=n)[0]]['loss_grad']
    df.max_grad = df.iloc[argrelextrema(df.loss_grad.values, np.greater_equal, order=n)[0]]['loss_grad']

    # Optional: Display dataframe if show_df=True
    if show_df == 'head': print(df.head(50)) 
    elif show_df == 'tail': print(df.tail(50))     
        
    # Plot losses and loss gradients against lr values
    plt.figure(figsize=[8, 5])
    #figs, ax = plt.subplots(1,1)
    ax = plt.gca()
    color_loss = 'blue'
    color_grad = 'orange'
    color_green = 'green'
    color_red = 'red'

    ax.xaxis.grid(True)
    ax.set_ylabel('Loss')
    ax.set_title('Learn Rate Finder')
    ax.tick_params(axis='y', labelcolor=color_loss)
    ax.semilogx(df.lrs, df.loss, c=color_loss, label='loss' )
    
    # Define variable vertical size of the plot window, depending on the graph shape
    u_limit = max(df.loss.loc[(df.lrs < 0.1)].max(), 250)*2    
    ax.set_ylim([-200, u_limit])
   
    # Plot resulting line graphs
    ax2 = ax.twinx()
    ax2.set_ylabel('loss', color= color_grad)
    ax2.semilogx(df.lrs, df.loss_grad, c = color_grad, label='loss_grad' )
    ax2.tick_params(axis='y', labelcolor = color_grad)
    
    # plot inflection points
    ax.scatter(df.lrs, df.min_loss, c = color_red, label='min_loss' )    
    ax2.scatter(df.lrs, df.min_grad, c = color_red, label='min_grad' )    
    if show_min_values:
        ax.scatter(df.lrs, df.max_loss, c = color_green, label='max_loss' )
        ax2.scatter(df.lrs, df.max_grad, c = color_green, label='max_grad' ) 
    
    # Legends
    plt.LogFormatter(labelOnlyBase=False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, fancybox=True, shadow=True)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 0.9), ncol=3, fancybox=True, shadow=True)
    plt.show()
    
    # Display resulting lr values, format varies depending of the state of the model's 
    # next-to-last layer ggroup: set to freeze or unfreeze    
    if is_unfreeze(learner):
        # Yellow min_grad graph
        rev_tru_idx = df.min_grad.notna()[::-1]   
        optimun_lr_upper_bound_g = df.lrs.iloc[rev_tru_idx.idxmax()] 
        rev_tru_idx[rev_tru_idx.idxmax()] = np.NaN      
        optimun_lr_lower_bound_1_g = df.lrs.iloc[rev_tru_idx.idxmax()]
        rev_tru_idx[rev_tru_idx.idxmax()] = np.NaN      
        optimun_lr_lower_bound_2_g = df.lrs.iloc[rev_tru_idx.idxmax()] 

        # Blue loss graph
        rev_tru_idx_loss = df.min_loss.notna()[::-1]   
        optimun_lr_upper_bound_l = df.lrs.iloc[rev_tru_idx_loss.idxmax()] 
        rev_tru_idx_loss[rev_tru_idx_loss.idxmax()] = np.NaN      
        optimun_lr_lower_bound_1_l = df.lrs.iloc[rev_tru_idx_loss.idxmax()]
        rev_tru_idx_loss[rev_tru_idx_loss.idxmax()] = np.NaN      
        optimun_lr_lower_bound_2_l = df.lrs.iloc[rev_tru_idx_loss.idxmax()] 

        # Print results and return choices of lr slice
        print('Model set to: "unfreeze" or "freeze_to:"')      
        data = {'*Gradient - Orange Graph*' : [optimun_lr_upper_bound_g, optimun_lr_lower_bound_1_g, optimun_lr_lower_bound_2_g], 
              '*Loss - Blue Graph*' : [optimun_lr_upper_bound_l, optimun_lr_lower_bound_1_l, optimun_lr_lower_bound_2_l]}
        prdf = pd.DataFrame(data, index = ['First choice lr:', 'Second choice lr:', 'Third choice lr:' ])
        pd.options.display.float_format = '{:.10E}'.format
        #prdf.style.applymap('color: %s' % color_grad, subset=['*Gradient - Orange Graph*'])
        print(prdf)

        return optimun_lr_lower_bound_1_g, optimun_lr_upper_bound_g
      
    else:
        
        optimun_lr_upper_bound = df.lrs.iloc[df.min_grad.notna()[::-1].idxmax()]
        optimun_lr_lower_bound = df.lrs.iloc[df.min_loss.notna()[::-1].idxmax()]/10
        # Print results and return optimal lr
        print('Model set to "freeze":')
        print('  Optimun lr: {:.10E} '.format(optimun_lr_upper_bound))
        print('  Min loss divided by 10: {:.10E}'.format(optimun_lr_lower_bound))
        return optimun_lr_upper_bound