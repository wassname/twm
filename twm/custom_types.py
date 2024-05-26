from typing import NewType, List, Dict, Annotated
from jaxtyping import Float, Int, Bool
from torch import Tensor, distributions as D

Obs = Float[Tensor, "b tgt_len frames odim"]
Z = Float[Tensor, "b tgt_len zdim"]
Hiddens = List[Float[Tensor, "b tgt_len hdim"]]
Logits = Float[Tensor, "b tgt_len z_categoricals z_categories"]
Z_dist = Float[D.Distribution, "b tgt_len z_categoricals z_categories"]
TrceInt = Int[Tensor, "b tgt_len"]
TrcFloat = Float[Tensor, "b tgt_len"]
TrcBool = Bool[Tensor, "b tgt_len"]
AA = Annotated[Tensor, "test"]
Actions = Float[Tensor, "b actions"]
# a
# r
