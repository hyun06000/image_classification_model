import numpy as np
import tensorflow as tf



def uint8_to_f32(i):
        return i.numpy().astype(np.float32)/255. * 2 -1

def data_whitening(train_dataset):
    
    for i, data in enumerate(train_dataset):
        images = data["image"]
        if not i:
            _data  = uint8_to_f32(images)
        else:
            _data += uint8_to_f32(images)
    
    return _data / (i + 1)


def train_loop(log_freq,
               train_log_dir,
               test_log_dir,
               tr_ds,
               te_ds,
               model,
               loss,
               optimizer,
               tr_accuracy,
               te_accuracy,
               EPOCH,
               TR_STEPS_PER_EPOCH,
               TE_STEPS_PER_EPOCH,
               hist_log,
               SCHEDULER
               ):
    
    mena_image = np.load('./asset/cifar10_mean_image.npy')
    
    # Tensorboard Writer
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    
    
    if log_freq == "epochs":
        
        for epochs in range(EPOCH):
            epochs += 1
            
            # Train loop
            loss_value, acc = 0, 0
            for tr_example in tr_ds:
                images, labels = tr_example["image"], tr_example["label"]
                with tf.GradientTape() as tape:
                    logits = model(uint8_to_f32(images) - mena_image, training = True)
                    for i, layer in enumerate(model.layers):
                        loss_value += tf.math.reduce_sum(layer.losses)
                    loss_value /= (i+1)
                    loss_value += loss(tf.one_hot(labels, 10), logits)

                    tr_accuracy.update_state(tf.one_hot(labels, 10), logits)
                    acc += tr_accuracy.result()
                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if SCHEDULER == 'custom':
                if epochs in [32000, 48000]:
                    optimizer.learning_rate = 0.1*optimizer.learning_rate.numpy()
            #Train log
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss_value / TR_STEPS_PER_EPOCH, step=epochs)
                tf.summary.scalar('acc', acc / TR_STEPS_PER_EPOCH , step=epochs)
                tr_accuracy.reset_state()
                
                if hist_log:
                    # histogram
                    for w in model.weights:
                        if "batch_normalization" in w.name:
                            tf.summary.histogram(
                                "batch_normalization/" + w.name, w, step=epochs)
                        elif "conv2d" in w.name:
                            tf.summary.histogram("conv2d/" + w.name, w, step=epochs)
                        elif "dense" in w.name:
                            tf.summary.histogram("dense/" + w.name, w, step=epochs)
                        else:
                            tf.summary.histogram(w.name, w, step=epochs)
            
            #Test loop
            loss_value, acc = 0, 0
            for te_example in te_ds:
                images, labels = te_example["image"], te_example["label"]
                logits = model.call(uint8_to_f32(images) - mena_image, training = False)
                loss_value += loss(tf.one_hot(labels, 10), logits)
                te_accuracy.update_state(tf.one_hot(labels, 10), logits)
                acc += te_accuracy.result()
            
            #test_log
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', loss_value / TE_STEPS_PER_EPOCH, step=EP)
                tf.summary.scalar('acc', acc / TE_STEPS_PER_EPOCH, step=EP)
                te_accuracy.reset_state()
    
    
    
    elif log_freq == "steps":
        
        step = 1
        for EP in range(EPOCH):
            EP += 1
            
            for tr_example, te_example in zip(tr_ds, te_ds):
                
                # Train loop
                loss_value, acc = 0, 0
                images, labels = tr_example["image"], tr_example["label"]
                with tf.GradientTape() as tape:
                    logits = model(uint8_to_f32(images) - mena_image, training = True)
                    for i, layer in enumerate(model.layers):
                        loss_value += tf.math.reduce_sum(layer.losses)
                    loss_value /= (i+1)
                    loss_value += loss(tf.one_hot(labels, 10), logits)
                    
                    tr_accuracy.update_state(tf.one_hot(labels, 10), logits)
                    acc += tr_accuracy.result()
                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                if SCHEDULER == 'custom':
                    if step in [32000, 48000]:
                        optimizer.learning_rate = 0.1*optimizer.learning_rate.numpy()
                
                
                #Train log
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss_value, step=step)
                    tf.summary.scalar('acc', acc, step=step)
                    tr_accuracy.reset_state()
                    
                    if hist_log:
                        # histogram
                        for w in model.weights:
                            if "batch_normalization" in w.name:
                                tf.summary.histogram(
                                    "batch_normalization/" + w.name, w, step=step)
                            elif "conv2d" in w.name:
                                tf.summary.histogram("conv2d/" + w.name, w, step=step)
                            elif "dense" in w.name:
                                tf.summary.histogram("dense/" + w.name, w, step=step)
                            else:
                                tf.summary.histogram(w.name, w, step=step)

                #Test loop
                loss_value, acc = 0, 0
                images, labels = te_example["image"], te_example["label"]
                logits = model.call(uint8_to_f32(images) - mena_image, training = False)
                loss_value = loss(tf.one_hot(labels, 10), logits)
                te_accuracy.update_state(tf.one_hot(labels, 10), logits)
                acc += te_accuracy.result()

                #test_log
                with test_summary_writer.as_default():
                    tf.summary.scalar('loss', loss_value, step=step)
                    tf.summary.scalar('acc', acc, step=step)
                    te_accuracy.reset_state()
                
                step += 1