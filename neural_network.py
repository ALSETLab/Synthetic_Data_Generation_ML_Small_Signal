def magical_loss(Z4, Y):
        
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = Z4, labels = Y)
    loss = tf.reduce_mean(loss)
    
    return loss

def NNEigenvalueClassification(X_train, Y_train, X_test, Y_test, X_train_full, Y_train_full,
                               learning_rate = 0.001, num_epochs = 50, minibatch_size = 500,
                              verbose = False):
    
    tf.reset_default_graph()
    
    # Getting dimensions of input tensor
    (m, n) = X_train.shape
    
    X = tf.placeholder(tf.float32, shape = [None, 2])
    Y = tf.placeholder(tf.float32, shape = [None, 6])
    if verbose:
        print(f"X = {X.shape}")
        print(f"Y = {Y.shape}")
    
    # First layer weights and biases   
    W1 = tf.get_variable('W1', shape = [2, 100], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    b1 = tf.Variable(tf.zeros(100))
    
    # Second layer weights and biases
    W2 = tf.get_variable('W2', shape = [100, 100], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    b2 = tf.Variable(tf.zeros(100))
    
    # Third layer weights and biases
    W3 = tf.get_variable('W3', shape = [100, 100], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    b3 = tf.Variable(tf.zeros(100))
    
    # Fourth layer weights and biases
    W4 = tf.get_variable('W4', shape = [100, 6], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    b4 = tf.Variable(tf.zeros(6))
    
    # === FIRST LAYER ===
    z1 = tf.add(tf.matmul(X, W1), b1)
    h1 = tf.nn.relu(z1)
    do1 = tf.layers.dropout(h1, rate = 0.25)
    if verbose:
        print(f"h1 = {h1.shape}")
    
    # === SECOND LAYER ===
    z2 = tf.add(tf.matmul(do1, W2), b2)
    h2 = tf.nn.relu(z2)
    do2 = tf.layers.dropout(h2, rate = 0.25)
    if verbose:
        print(f"h2 = {h2.shape}")
    
    # === THIRD LAYER ===
    z3 = tf.add(tf.matmul(do2, W3), b3)
    h3 = tf.nn.relu(z3)
    if verbose:
        print(f"h3 = {h3.shape}")
    
    # === OUTPUT LAYER ===
    z4 = tf.add(tf.matmul(h3, W4), b4)
    if verbose:
        print(f"z4 (logits) = {z4.shape}")
    
    # Computing loss function    
    loss = magical_loss(z4, Y)
    if verbose:
        print(f"loss = {loss.shape}")
    
    # Definining optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
    
    # ================================= EXECUTION ================================
    
    # Initializer for global variables
    init = tf.global_variables_initializer()
    
    # GPU Options
    gpu_options = tf.GPUOptions(allow_growth=True)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, \
                                          log_device_placement=True,gpu_options = gpu_options)) as sess:
        
        # Starting record of training time
        t_0 = time.time()
        
        # Print starting execution time
        st = datetime.datetime.fromtimestamp(t_0).strftime('%Y-%m-%d %H:%M:%S')
        print(st)
        
        # Computing the number of minibatches in the training set 
        num_minibatches = int(m/minibatch_size)        
        
        # Initializing variables of the computation graph
        sess.run(init)
        
        # Saver for exporting model
        saver = tf.train.Saver()
        
        # Containers for loss, training accuracy and testing accuracy
        loss_value = []
        train_accuracy_value = []
        test_accuracy_value = []
        
        print(f"Num of minibatches: {num_minibatches}")
        
         # Iterating through epochs
        for epoch in range(num_epochs):
            
            indices = np.arange(m)
            np.random.shuffle(indices)
            
            # Reset minibatch loss to zero
            loss_mb = 0
            
            # Iterating in minibatch
            for minibatch in range(num_minibatches):
                
                # Informative print statement
                if (minibatch % 500 == 0) and verbose:
                    print("Iterating in minibatch... ({}/{})".format(minibatch, num_minibatches))
                
                # Generating a minibatch
                X_mb = X_train[indices[minibatch*minibatch_size:(minibatch+1)*minibatch_size],...]
                Y_mb = Y_train[indices[minibatch*minibatch_size:(minibatch+1)*minibatch_size],...]
                if verbose:
                    print(f"X_mb = {X_mb.shape}")
                    print(f"Y_mb = {Y_mb.shape}")
                
                # Running optimizer and training
                _, temp_loss_mb = sess.run([optimizer, loss], feed_dict = {X: X_mb, Y: Y_mb})
                
                # Updating loss
                loss_mb += temp_loss_mb / num_minibatches
            
            # End of minibatch loop
            
            # Print loss after a given number of epochs
            if epoch % 10 == 0:
                print("Loss after epoch {} = {}".format(epoch + 1, loss_mb))
            
            loss_value.append(loss_mb)
            
            predict_op = tf.argmax(z4, 1)
            correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

            # Defining graph for accuracy using actual values of weights      
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            #Computing training and testing accuracies
            train_accuracy_value.append(accuracy.eval({X: X_train, Y: Y_train}))
            test_accuracy_value.append(accuracy.eval({X: X_test, Y: Y_test}))    
            
        # End of training iterations
        t_f = time.time() - t_0
        print("\nTRAINING FINISHED\n")
        print(f"Training time: {t_f:.3f} s")
        print(f"Training accuracy: {train_accuracy_value[-1]}")
        print(f"Test accuracy: {test_accuracy_value[-1]}\n")
        
        ####################
        #### PREDICTION ####
        ####################
        
        predict_op = tf.argmax(z4, 1)                                     
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Computing predictions for training set
        output_NN_train = sess.run(predict_op, {X: X_train, Y: Y_train})
        # Computing predictions for testing set
        output_NN_test = sess.run(predict_op, {X: X_test, Y: Y_test})
        
        # Defining graph for accuracy using actual values of weights      
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        # Computing training and testing accuracie
        final_accuracy_training = accuracy.eval({X: X_train, Y: Y_train})
        
        t_0 = time.time()
        final_accuracy_training_full = accuracy.eval({X: X_train_full, Y: Y_train_full})
        final_accuracy_testing = accuracy.eval({X: X_test, Y: Y_test})
        
        print("Elapsed time (prediction): {t_f:.3f} s".format(t_f = time.time() - t_0))
        
        print(f"Training accuracy: {final_accuracy_training}")
        print(f"Test accuracy: {final_accuracy_testing}")
        
        tf.get_collection('validation_nodes', predict_op)
        
        # Saving model
        save_path = saver.save(sess, "99_Data/my_model")
        
        ts = time.time()
        et = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        print(et)
        
    # Exporting training results
    
    with open("99_Data/output_NN_train.pkl", 'wb') as f:
        pickle.dump(output_NN_train, f, pickle.HIGHEST_PROTOCOL)
    
    with open("99_Data/output_NN_test.pkl", 'wb') as f:
        pickle.dump(output_NN_test, f, pickle.HIGHEST_PROTOCOL)
    
    # Saving training accuracy info
    with open("99_Data/train_accuracy_value.pkl", 'wb') as f:
        pickle.dump(train_accuracy_value, f, pickle.HIGHEST_PROTOCOL)
    
    # Saving testing accuracy info
    with open("99_Data/test_accuracy_value.pkl", 'wb') as f:
        pickle.dump(test_accuracy_value, f, pickle.HIGHEST_PROTOCOL)
    
    # Saving loss function info
    with open("99_Data/loss_value.pkl", 'wb') as f:
        pickle.dump(loss_value, f, pickle.HIGHEST_PROTOCOL)
        
def plot_nn_performance_info():
    with open("99_Data/train_accuracy_value.pkl", 'rb') as f:
        train_accuracy_value = pickle.load(f)
    
    with open("99_Data/test_accuracy_value.pkl", 'rb') as f:
        test_accuracy_value = pickle.load(f)

    with open("99_Data/loss_value.pkl", 'rb') as f:
        loss_value = pickle.load(f)

    # Plot 1: Loss versus number of epochs
    plt.figure(1)
    plt.plot(np.squeeze(loss_value))
    plt.xlabel("No. of Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Function")
    plt.savefig('Figs/LossFunction.png')

    # Plot 2: Training and Testing Accuracy    
    plt.figure(2)
    plt.plot(np.squeeze(train_accuracy_value), label = 'Training Acc', color = 'indigo')
    plt.plot(np.squeeze(test_accuracy_value), label = 'Testing Acc', color = 'salmon')
    plt.xlabel("No. of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Testing Accuracy")
    plt.legend()
    plt.savefig('Figs/TrainingAndTestingAccuracy.png')

    # Plot 3: Loss, Training and Testing Accuracy in one figure

    plt.figure(3)
    fig, ax1 = plt.subplots();
    ax2 = ax1.twinx()

    ax1.plot(np.squeeze(loss_value), label = 'Loss', color = 'royalblue')
    ax1.set_xlabel("No. of Epochs", fontname = 'liberation sans')
    ax1.set_ylabel("Loss", fontname = 'liberation sans')

    ax2.plot(np.squeeze(train_accuracy_value), label = 'Training Acc', color = 'indigo')
    ax2.plot(np.squeeze(test_accuracy_value), label = 'Testing Acc', color = 'salmon')
    ax2.set_ylabel("Accuracy", fontname = 'liberation sans')
    ax2.set_ylim([0, 1])

    ax1.legend(loc = 'center right', prop = {'family' : 'liberation sans'})
    ax2.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5), prop = {'family' : 'liberation sans'})
    # Formatting ticks
    for tick in ax1.get_xticklabels():
        tick.set_fontname('liberation sans')
        tick.set_fontsize(13)
    for tick in ax1.get_yticklabels():
        tick.set_fontname('liberation sans')
        tick.set_fontsize(13)
    for tick in ax2.get_xticklabels():
        tick.set_fontname('liberation sans')
        tick.set_fontsize(13)
    for tick in ax2.get_yticklabels():
        tick.set_fontname('liberation sans')
        tick.set_fontsize(13)

    plt.title("Loss, Training and Testing Accuracy", fontname = 'liberation sans')
    plt.savefig('Figs/TrainingAndTestingAccuracy.png')