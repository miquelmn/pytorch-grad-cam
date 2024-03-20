import torch
import tqdm

from pytorch_grad_cam.base_cam import BaseCAM


class ScoreCAM(BaseCAM):
    def __init__(
            self,
            model,
            target_layers,
            use_cuda=False,
            reshape_transform=None,
            show_progress=True,
            batch_size=16,
            *args,
            **kwargs):
        self.show_progress = show_progress
        self.batch_size = batch_size
        super(ScoreCAM, self).__init__(model,
                                       target_layers,
                                       use_cuda,
                                       reshape_transform=reshape_transform,
                                       uses_gradients=False,
                                       *args,
                                       **kwargs)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        targets,
                        activations,
                        grads):
        with torch.no_grad():
            upsample = torch.nn.UpsamplingBilinear2d(
                size=input_tensor.shape[-2:])
            activation_tensor = torch.from_numpy(activations)
            if self.cuda:
                activation_tensor = activation_tensor.cuda(self.device)

            upsampled = upsample(activation_tensor)

            maxs = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).max(dim=-1)[0]
            mins = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).min(dim=-1)[0]

            maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
            upsampled = (upsampled - mins) / ((maxs - mins) + 1e-5)

            input_tensors = input_tensor[:, None, :, :] * upsampled[:, :, None, :, :]

            scores = []
            for target, tensor in zip(targets, input_tensors):
                for i in tqdm.tqdm(range(0, tensor.size(0), self.batch_size),
                                   disable=not self.show_progress):
                    batch = tensor[i: i + self.batch_size, :]
                    outputs = [target(o).cpu().item()
                               for o in self.model(batch.cuda(self.device))]
                    scores.extend(outputs)
            scores = torch.Tensor(scores)
            scores = scores.view(activations.shape[0], activations.shape[1])
            weights = torch.nn.Softmax(dim=-1)(scores).numpy()
            return weights
