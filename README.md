# FDM: Fair Diversity Maximization in Streaming and Sliding-Window Models
In this repository, we implement **FDM** in Streaming and Sliding-Window Models. Algorithms(Fair-Swap, Fair-Flow and Fair-GMM) in [**Diverse Data Selection under Fairness Constraints**](https://arxiv.org/pdf/2010.09141.pdf) and Fair-Greedy-Flow in [**Improved Approximation and Scalability for Fair Max-Min Diversification**](https://arxiv.org/pdf/2201.06678.pdf) are also implemented, which are treated as baseline.

## Datasets

Four real-world datasets ,i.e. Adult, CelebA, Census, Lyrics and one sythetic dataset are used in our experiments. Download links are included in the paper. To generate sythetic dataset, run the command like the following :
```python
python generate_synthetic_dataset.py -n 1000 -m 2 -d 2
```

<table border='0' cellpadding='0' cellspacing='0' width='682' style='border-collapse: 
 collapse;t<col class='x21' width='176' style='mso-width-source:userset;width:132pt'>
 <col class='x21' width='61' style='mso-width-source:userset;width:45.75pt'>
 <col class='x21' width='77' style='mso-width-source:userset;width:57.75pt'>
 <col class='x21' width='192' style='mso-width-source:userset;width:144pt'>
 <tr height='26' style='mso-height-source:userset;height:20pt' id='r0'>
<td height='26' class='x22' width='97' style='height:20pt;width:72.75pt;'>Dataset</td>
<td class='x22' width='79' style='width:59.25pt;'>Group</td>
<td class='x22' width='176' style='width:132pt;'>n </td>
<td class='x22' width='61' style='width:45.75pt;'>m </td>
<td class='x21' width='77' style='width:57.75pt;'><font class="font3">#</font><font class="font4">feats </font></td>
<td class='x22' width='192' style='width:144pt;'>Distance Metric </td>
 </tr>
 <tr height='26' style='mso-height-source:userset;height:20pt' id='r1'>
<td rowspan='3' height='80' class='x26' style='height:60pt;'>Adult </td>
<td class='x21'>Sex</td>
<td rowspan='3' height='80' class='x26' style='height:60pt;'>48, 842 </td>
<td class='x21'>2</td>
<td rowspan='3' height='80' class='x26' style='height:60pt;'>6</td>
<td rowspan='3' height='80' class='x26' style='height:60pt;'>Euclidean </td>
 </tr>
 <tr height='26' style='mso-height-source:userset;height:20pt' id='r2'>
<td class='x21'>Race</td>
<td class='x21'>5</td>
 </tr>
 <tr height='26' style='mso-height-source:userset;height:20pt' id='r3'>
<td class='x21'>S+R </td>
<td class='x21'>10</td>
 </tr>
 <tr height='28' style='mso-height-source:userset;height:21pt' id='r4'>
<td rowspan='3' height='81' class='x26' style='height:61pt;'>CelebA </td>
<td class='x21'>Sex </td>
<td rowspan='3' height='81' class='x27' style='height:61pt;'>202, 599</td>
<td class='x21'>2</td>
<td rowspan='3' height='81' class='x26' style='height:61pt;'>41</td>
<td rowspan='3' height='81' class='x26' style='height:61pt;'>Manhattan </td>
 </tr>
 <tr height='26' style='mso-height-source:userset;height:20pt' id='r5'>
<td class='x21'>Age</td>
<td class='x21'>2</td>
 </tr>
 <tr height='26' style='mso-height-source:userset;height:20pt' id='r6'>
<td class='x21'>S+A </td>
<td class='x21'>4</td>
 </tr>
 <tr height='26' style='mso-height-source:userset;height:20pt' id='r7'>
<td rowspan='3' height='80' class='x26' style='height:60pt;'>Census </td>
<td class='x21'>Sex </td>
<td rowspan='3' height='80' class='x26' style='height:60pt;'>2, 426, 116 </td>
<td class='x21'>2</td>
<td rowspan='3' height='80' class='x28' style='height:60pt;'>25</td>
<td rowspan='3' height='80' class='x26' style='height:60pt;'>Manhattan </td>
 </tr>
 <tr height='26' style='mso-height-source:userset;height:20pt' id='r8'>
<td class='x21'>Age</td>
<td class='x21'>7</td>
 </tr>
 <tr height='26' style='mso-height-source:userset;height:20pt' id='r9'>
<td class='x21'>S+A </td>
<td class='x21'>14</td>
 </tr>
 <tr height='26' style='mso-height-source:userset;height:20pt' id='r10'>
<td height='26' class='x21' style='height:20pt;'>Lyrics</td>
<td class='x21'>Genre</td>
<td class='x21'>122, 448 </td>
<td class='x21'>15</td>
<td class='x21'>15</td>
<td class='x21'>Angular </td>
 </tr>
 <tr height='26' style='mso-height-source:userset;height:20pt' id='r11'>
<td height='26' class='x21' style='height:20pt;'>SYN </td>
<td class='x23'>-</td>
<td class='x24'>1000-10,000,000</td>
<td class='x25'>2-20</td>
<td class='x21'>2</td>
<td class='x21'>Euclidean </td>
 </tr>
</table>


## Requirements

- Ubuntu 20.04 (or higher version)
- Python 3.8 (or higher version)

## Experiments

Once you have downloaded the datasets and put them in the  directory *datasets*, you can reproduce the experiments by simply running the following commands:

```python
python run_exp_varying_k_m_n_stream.py
python run_exp_varying_k_window.py 
python run_exp_varying_m_w_window.py
```
