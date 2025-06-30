from typing import Any, Dict, List, Mapping, Optional

import torch
from torch import nn

from video_SA.videosaur.utils import make_build_fn
import model.model_operator_slots as slots_op

@make_build_fn(__name__, "video module")
def build(config, name: str, **kwargs):
    pass  # No special module building needed


class LatentProcessor(nn.Module):
    """Updates latent state based on inputs and state and predicts next state."""

    def __init__(
        self,
        corrector: nn.Module,
        predictor: Optional[nn.Module] = None,
        state_key: str = "slots",
        first_step_corrector_args: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.corrector = corrector
        self.predictor = predictor
        self.state_key = state_key
        if first_step_corrector_args is not None:
            self.first_step_corrector_args = first_step_corrector_args
        else:
            self.first_step_corrector_args = None

    def forward(
        self, state: torch.Tensor, inputs: Optional[torch.Tensor], time_step: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        # state: batch x n_slots x slot_dim
        assert state.ndim == 3
        # inputs: batch x n_inputs x input_dim
        assert inputs.ndim == 3

        if inputs is not None:
            if time_step == 0 and self.first_step_corrector_args:
                corrector_output = self.corrector(state, inputs, **self.first_step_corrector_args)
            else:
                corrector_output = self.corrector(state, inputs)
            updated_state = corrector_output[self.state_key]
        else:
            # Run predictor without updating on current inputs
            corrector_output = None
            updated_state = state

        if self.predictor:
            predicted_state = self.predictor(updated_state)
        else:
            # Just pass updated_state along as prediction
            predicted_state = updated_state

        return {
            "state": updated_state,
            "state_predicted": predicted_state,
            "corrector": corrector_output,
        }


class MapOverTime(nn.Module):
    """Wrapper applying wrapped module independently to each time step.

    Assumes batch is first dimension, time is second dimension.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args):
        batch_size = None
        seq_len = None
        flattened_args = []
        for idx, arg in enumerate(args):
            B, T = arg.shape[:2]
            if not batch_size:
                batch_size = B
            elif batch_size != B:
                raise ValueError(
                    f"Inconsistent batch size of {B} of argument {idx}, was {batch_size} before."
                )

            if not seq_len:
                seq_len = T
            elif seq_len != T:
                raise ValueError(
                    f"Inconsistent sequence length of {T} of argument {idx}, was {seq_len} before."
                )

            flattened_args.append(arg.flatten(0, 1))

        outputs = self.module(*flattened_args)

        if isinstance(outputs, Mapping):
            unflattened_outputs = {
                k: v.unflatten(0, (batch_size, seq_len)) for k, v in outputs.items()
            }
        else:
            unflattened_outputs = outputs.unflatten(0, (batch_size, seq_len))

        return unflattened_outputs

class MapOverTime2(nn.Module):
    """Wrapper applying wrapped module independently to each time step.

    Assumes batch is first dimension, time is second dimension.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args):
        batch_size = None
        seq_len = None
        flattened_args = []
        
        for idx, arg in enumerate(args):
            B, T = arg.shape[:2]
            if batch_size is None:
                batch_size = B
            elif batch_size != B:
                raise ValueError(
                    f"Inconsistent batch size of {B} for argument {idx}, was {batch_size} before."
                )

            if seq_len is None:
                seq_len = T
            elif seq_len != T:
                raise ValueError(
                    f"Inconsistent sequence length of {T} for argument {idx}, was {seq_len} before."
                )

            flattened_args.append(arg.flatten(0, 1))

        outputs = self.module(*flattened_args)

        # Ensure outputs is always a tuple for consistent processing
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        # Unflatten each output
        unflattened_outputs = tuple(
            output.unflatten(0, (batch_size, seq_len)) for output in outputs
        )

        # Return a single output if only one exists; otherwise, return the tuple
        return unflattened_outputs if len(unflattened_outputs) > 1 else unflattened_outputs[0]
class MapOverTime_mask(nn.Module):
    """Wrapper applying wrapped module independently to each time step.

    Assumes batch is the first dimension, time is the second dimension.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, args, args2):
        # Process args (shape [Batch, T, S, D])
        B, T, S, D = args.shape
        flattened_args = args.flatten(0, 1)  # Flatten Batch and T dimensions.

        # Process args2 (shape [Batch, T, S])
        if args2 is not None:
            B2, T2, S2 = args2.shape
            if B != B2 or T != T2 or S != S2:
                raise ValueError(
                    f"Inconsistent dimensions between args {args.shape} and args2 {args2.shape}."
                )
            flattened_args2 = args2.flatten(0, 1)  # Flatten Batch and T dimensions.
        else:
            flattened_args2 = None

        # Pass the flattened inputs to the module
        outputs = self.module(flattened_args, flattened_args2)

        # Unflatten outputs back to [Batch, T, ...]
        if isinstance(outputs, Mapping):
            unflattened_outputs = {
                k: v.unflatten(0, (B, T)) for k, v in outputs.items()
            }
        else:
            unflattened_outputs = outputs.unflatten(0, (B, T))

        return unflattened_outputs
import torch.nn as nn
from collections.abc import Mapping
class IterOverTime_mask(nn.Module):
    """Wrapper applying wrapped module iteratively to each time step.

    Assumes batch is the first dimension, time is the second dimension.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, args, args2=None):
        # Extract dimensions from args (shape [Batch, T, S, D])
        B, T, S, D = args.shape

        # Validate args2 dimensions if provided
        if args2 is not None:
            B2, T2, S2 = args2.shape
            if B != B2 or T != T2 or S != S2:
                raise ValueError(
                    f"Inconsistent dimensions between args {args.shape} and args2 {args2.shape}."
                )

        outputs = []

        # Iterate over each time step
        for t in range(T):
            # Slice args and args2 at current time step
            time_slice_args = args[:, t, :, :]  # Shape: [B, S, D]
            time_slice_args2 = args2[:, t, :] if args2 is not None else None  # Shape: [B, S]

            # Apply module to the time slice
            output = self.module(time_slice_args, time_slice_args2)
            outputs.append(output)

        # Combine outputs across time steps
        if isinstance(outputs[0], Mapping):
            unflattened_outputs = {
                k: torch.stack([out[k] for out in outputs], dim=1) for k in outputs[0].keys()
            }
        else:
            unflattened_outputs = torch.stack(outputs, dim=1)  # Stack along time dimension

        return unflattened_outputs
class IterOverTime(nn.Module):
    """Wrapper applying wrapped module independently to each time step.

    Assumes batch is the first dimension, time is the second dimension.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args):
        batch_size = None
        seq_len = None
        outputs = []

        for t in range(args[0].shape[1]):  # Iterate over time dimension
            time_slice = [arg[:, t] for arg in args]  # Slice each argument at time t

            if not batch_size:
                batch_size = time_slice[0].shape[0]

            # Apply the module to the time slice
            output = self.module(*time_slice)
            outputs.append(output)

        if isinstance(outputs[0], Mapping):
            # Combine the outputs for each time step
            unflattened_outputs = {
                k: torch.stack([out[k] for out in outputs], dim=1) for k in outputs[0].keys()
            }
        else:
            unflattened_outputs = torch.stack(outputs, dim=1)  # Stack over time dimension

        return unflattened_outputs

class ScanOverTime(nn.Module):
    """Wrapper applying wrapped module recurrently over time steps"""

    def __init__(
        self, module: nn.Module, next_state_key: str = "state_predicted", pass_step: bool = True
    ) -> None:
        super().__init__()
        self.module = module
        self.next_state_key = next_state_key
        self.pass_step = pass_step

    def forward(self, initial_state: torch.Tensor, inputs: torch.Tensor,rnn=True,Next_state_predict=None,model=None,merger=None):
        # initial_state: batch x ...
        # inputs: batch x n_frames x ...
        seq_len = inputs.shape[1]

        state = initial_state #B,S,D_slot
        next_state = state*0
        slot_state_buff =  []
        B, S, D_slot= initial_state.shape
        outputs = []
        for t in range(seq_len):
            if rnn==False:
                state = initial_state # if stoped the RNN propgate
                # create bufer of staste  
            if self.pass_step:
                output = self.module(state, inputs[:, t], t)
                # output["state"] = 0.5 * output["state"] + 0.5 * state
            else:
                output = self.module(state, inputs[:, t])
            outputs.append(output)
            # state = output[self.next_state_key]
            if Next_state_predict is None:
                state = output["state"]
            elif Next_state_predict == "videosaur":
                state = output[self.next_state_key]
            elif Next_state_predict == "binder" or Next_state_predict == "binder+merger":
                state = output["state"]
                if model is not None:
                    
                    if t ==0:
                        for _ in range(3):
                            slot_state_buff.append(state)
                    else:
                        # B F, S, D_slot  
                        slot_state_buff.append(state)
                        # slot_state_buff .pop(0) # remove the oldest
                        
                        slot_state_buff.append(initial_state*0.0)
                        slot_state_window= torch.stack (slot_state_buff)
                        slot_state_window=slot_state_window.permute (1,0,2,3)# B F, S, D_slot  
                        slotst,slots = model(slot_state_window,usingmask=False)  # B F, S, D_slot  
                        state = slots[:,4,:,:]
                        slot_state_buff .pop(4) # remove the added  
                        slot_state_buff .pop(0) # remove the oldest
                if Next_state_predict == "binder+merger":
                    state,slot_mask = merger(state)
                    # state[:,1,:]= state[:,6,:]
                    # state[:,2,:]= state[:,6,:]
                    # state[:,3,:]= state[:,6,:]

             


            # state = slots_op.add_noise_to_latents(state,noise_std=0.05)

        return merge_dict_trees(outputs, axis=1)


def merge_dict_trees(trees: List[Mapping], axis: int = 0):
    """Stack all leafs given a list of dictionaries trees.

    Example:
    x = merge_dict_trees([
        {
            "a": torch.ones(2, 1),
            "b": {"x": torch.ones(2, 2)}
        },
        {
            "a": torch.ones(3, 1),
            "b": {"x": torch.ones(1, 2)}
        }
    ])

    x == {
        "a": torch.ones(5, 1),
        "b": {"x": torch.ones(3, 2)}
    }
    """
    out = {}
    if len(trees) > 0:
        ref_tree = trees[0]
        for key, value in ref_tree.items():
            values = [tree[key] for tree in trees]
            if isinstance(value, torch.Tensor):
                out[key] = torch.stack(values, axis)
            elif isinstance(value, Mapping):
                out[key] = merge_dict_trees(values, axis)
            else:
                out[key] = values

    return out
