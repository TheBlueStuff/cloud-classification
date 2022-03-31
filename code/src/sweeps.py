from . dataset import GCD
from . import engine_gnn as engine
from . import utils
from sklearn.metrics import accuracy_score

"""
W&B sweeps
"""

def train(config=None):

#Init new run
    with wandb.init(config=config):

        config = wandb.config
        train_loader, 
        test_loader, 
        augmentation_loaders = utils.build_dataset(config.batch_size)

        model = utils.build_model_gatconv(
                                            7, #GCD num classes
                                            config.hidden_dim,
                                            config.num_hidden,
                                            config.num_heads,
                                            config.similarity_threshold,
                                           )

        optimizer = utils.build_optimizer(config.optimizer, model, config.learning_rate)
        criterion = utils.build_criterion(config.criterion)

        for e in range(config.epochs):
            ### TRAIN DATASET
                preds, 
                targets, 
                loss = engine.train_fn(model, 
                                       train_loader, 
                                       criterion, 
                                       optimizer, 
                                       config.device)
                
                train_acc = accuracy_score(targets, preds)

                ### AUGMENTATION IMAGES
                for _loader in augmentation_loaders:
                    engine.forward_backward_pass(model, 
                                                 _loader, 
                                                 criterion, 
                                                 optimizer, 
                                                 device=config.device)

                test_preds, 
                test_targets, 
                test_loss = engine.eval_fn(model, 
                                           test_loader, 
                                           criterion, 
                                           device=config.device)
                test_acc = accuracy_score(test_targets, test_preds)


                print("EPOCH {}: Train acc: {:.2%} Train Loss: {:.4f} Test acc: {:.2%} Test Loss: {:.4f}".format(
                    e+1,
                    train_acc,
                    loss,
                    test_acc,
                    test_loss
                ))

                metrics = {
                            "train/train_loss": loss,
                            "train/train_accuracy": train_acc,
                            "test/test_loss": test_loss,
                            "test/test_accuracy": test_acc,
                          }

                wandb.log(metrics)
            
        wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None, 
                                                            preds=test_preds,
                                                            y_true=test_targets,
                                                            class_names=classes)})
