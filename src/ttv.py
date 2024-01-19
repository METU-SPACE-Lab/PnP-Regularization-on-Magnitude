# General purpose training testing and validation (ttv) codes
from tqdm import tqdm
import numpy as np
import torch

import gc
from src.utils.data import make_dir,pickle_dump


def reconstruct_all(reconstructor, data_loader, device):
    reconstructor = reconstructor.to(device)
    reconstructions = []

    for meas, target in tqdm(data_loader):
        # Send the data to device
        meas=meas.to(device=device)
        target=target.to(device=device)

        # Reconstruct from measurements
        reconstruction = reconstructor(meas)
        
        # Log the reconstruction
        reconstructions.append(reconstruction.to(torch.device('cpu')))
    
    # return the reconsturcitons as a torch tensor
    return torch.cat(reconstructions, dim=0)

def evaluate_performance(reconstructor, data_loader, device, **performance_criteria):

    # Create a dictionary to store performances
    performances={}
    for perf_name, perf_func in performance_criteria.items():
        performances[perf_name]=[]

    # Send the reconstructor to device
    reconstructor  = reconstructor.to(device)

    for meas, target in tqdm(data_loader):
        
        # Send the data to device
        meas=meas.to(device=device)
        target=target.to(device=device)

        # Reconstruct from measurements
        reconstruction = reconstructor(meas)
        
        # Compute performance for each criterion
        for perf_name, perf_func in performance_criteria.items():
            perf_val = perf_func(estimation=reconstruction,target=target)
            
            performances[perf_name].append( np.reshape(np.squeeze(perf_val.cpu().detach().numpy()),(-1,1))) 

    for perf_name, perf_func in performance_criteria.items():
        performances[perf_name]=np.concatenate(performances[perf_name],axis=0)

    return performances

class EarlyStopWrapper():
    def __init__(self, tolerance=7, checkpoint_log='each', path=""):

        self.checkpoint_log=checkpoint_log
        assert (self.checkpoint_log == 'each') or (self.checkpoint_log == 'last') or (self.checkpoint_log == 'none'), \
            'Checkpoint log can be "each", "last" or "none' \
            '\nSet Checkpoint log to "each" to dump each check point,'\
            '\nSet Checkpoint log to "last" to dump the last check point.' \
            '\nSet Checkpoint log to "none" to only return True to break the training loop' \
            '\nSet Checkpoint log to "last" to dump the last check point.' \
            '\n "each" does not hold the passed checkpoint in cpu whereas "last" does.' \
            ' "Each" writes onto the disk at each new best checkpoint.'


        self.tolerance = tolerance
        self.checkpoint = False
        self.path = path
        self.counter = 0
        self.loss_min=np.infty
        self.call_idx = 0
        self.early_stop_call_idx = 0


        if not self.path.endswith('/'):
            self.path=self.path+'/'

    def _logCheckpoint(self, checkpoint, name):
        if self.checkpoint_log == 'each':
            pickle_dump(checkpoint, self.path+name)
        elif self.checkpoint_log == 'last':
            self.checkpoint=checkpoint
        elif self.checkpoint_log=='none':
            pass

    def _saveCheckpoint(self, checkpoint, name):
        if self.checkpoint_log == 'each':
            pass
        elif self.checkpoint_log == 'last':
            pickle_dump(checkpoint, self.path+name)
        elif self.checkpoint_log=='none':
            pass

    def earlyStop(self, loss,checkpoint,name):
        if loss > self.loss_min:
            self.counter = self.counter+1
        else:
            self.loss_min=loss
            self.early_stop_call_idx = self.call_idx
            self.counter=0
            self._logCheckpoint(checkpoint,name)

        self.call_idx=self.call_idx+1

        if self.counter > self.tolerance:
            self._saveCheckpoint(checkpoint,name)
            return True
        else:
            return False


    def reset(self):
        self.checkpoint = False
        self.counter = 0
        self.loss_min = np.infty
        self.call_idx = 0
        self.early_stop_call_idx = 0

def train(reconstructor, optimizer, loss_f,
                   train_loader, val_loader,
                   nof_epochs, path, device=torch.device('cpu'), scheduler=None,
                   earlyStopWrapper=EarlyStopWrapper(tolerance=10,checkpoint_log='each',path='')):

    gc.collect()
    if device != torch.device('cpu'):
        torch.cuda.empty_cache()
    reconstructor=reconstructor.to(device)

    if not path.endswith('/'):
        path = path + '/'

    make_dir(path)
    earlyStopWrapper.path=path
    early_stop_bool=False

    # Logging the losses
    train_loss = torch.zeros(size=(nof_epochs,), dtype=torch.float, device=device)  # Training losses averaged on epochs
    val_loss = torch.zeros(size=(nof_epochs,), dtype=torch.float, device=device)  # validation losses averaged on epochs


    loss_f.reduction = 'mean' # loss reduction should be set to mean so that gradients are acumulated as avarages, this 
    # prevents opt.steps being dependent of batch size. If  reduction was set to 'sum' gradient accumulation would be -
    # proportional to the batch size which in turn could make the training process unstable.

    # Start Training
    for epoch in tqdm(range(nof_epochs)):
    
        # Stop training if Early Stopping is Activated
        if early_stop_bool==True:
            tqdm.write(f'Early Stopping @ Epoch{epoch}\n')
            break

        #tqdm.write(f" @ EPOCH: {epoch}")
        #tqdm.write("----------------------------\nTraining\n----------------------------")
        
        # set nn module to training mode
        reconstructor.train()

        for train_in, train_target in train_loader:

            # Send data to device
            train_in = train_in.to(device)
            train_target = train_target.to(device)

            # Forward pass / estimate
            est = reconstructor(train_in)

            # BACK PROP
            loss = loss_f(est, train_target)  # Compute Loss
            train_loss[epoch] += loss.detach().item()*train_target.shape[0]  # Save the loss by accumulating it per epoch
            # We are multiplying with the batch size since loss reduction is averageing

            loss.backward()  # Backward pass
            optimizer.step()  # Optimization step
            optimizer.zero_grad()  # Zero gradients

        # After the loss is accumulated on all batches within the epoch save the average loss
        train_loss[epoch] = train_loss[epoch]/len(train_loader.dataset) # Normalize the loss

        #tqdm.write("----------------------------\nEvaluation\n----------------------------")
        
        # set nn module to training mode
        reconstructor.eval()
        
        # disable/freeze auto-grad
        with torch.no_grad():

            for val_in, val_target in val_loader:
                # Send to device
                val_in = val_in.to(device)
                val_target = val_target.to(device)

                # Forward pass
                est = reconstructor(val_in)

                # Compute Loss
                val_loss[epoch] += loss_f(est, val_target)*val_target.shape[0]# Save the loss by accumulating it per epoch
                # We are multiplying with the batch size since loss reduction is averageing
        
        # After the loss is accumulated on all batches within the epoch save the average loss
        val_loss[epoch] = val_loss[epoch]/len(val_loader.dataset)
        
        # Update optimization algorithm's parameters with the scheduler
        if scheduler != None:
            scheduler.step(val_loss[epoch])

        # Zero the gradients (NOTE: eval() mode disables stochastic behaviour of the nn and no_grad() disables 
        # gradients for the leaf variables nevertheless params.requires_grad stays true for nn params. Therefore I am
        # setting the gradients explicitly zero just incase. This might not be strictly necessary.)
        optimizer.zero_grad()  # Zero gradients

        # Construct the checkpoint
        checkpoint = { 'reconstructor':reconstructor,
                        'optimizer':optimizer,
                        'earlystop_epoch_idx':epoch,
                        'scheduler':scheduler,
                        'val_loss':val_loss[0:epoch+1].cpu().numpy(),
                        'train_loss':train_loss[0:epoch+1].cpu().numpy()}

        # Send the checkpoint to the early stopper
        early_stop_bool = earlyStopWrapper.earlyStop(loss=val_loss[epoch],
                                                     checkpoint=checkpoint,
                                                     name='checkpoint.pkl')

    # save the training losses at the end of the Training
    np.save( path+'train_loss_EOT.npy',train_loss[0:epoch].cpu().numpy())
    np.save( path+'val_loss_EOT.npy',val_loss[0:epoch].cpu().numpy())
    

    




