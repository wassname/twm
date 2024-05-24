from typing import NewType, List, Dict, Annotated
from jaxtyping import Float, Int, Bool
from torch import Tensor, distributions as D

Obs = Float[Tensor, 'b tgt_len odim']
Z = Float[Tensor, 'b tgt_len z']
Logits = Float[Tensor, 'b tgt_len z_categoricals z_categories']
Z_dist = Float[D.Distribution, 'b tgt_len z_categoricals z_categories']
TrceInt = Int[Tensor, 'b tgt_len']
TrcFloat = Float[Tensor, 'b tgt_len']
TrcBool = Bool[Tensor, 'b tgt_len']
Hiddens = List[Float[Tensor, 'b tgt_len d']]
AA = Annotated[Tensor, 'test']
# a
# r
