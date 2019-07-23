# FixRes

<img src="../image/image2.png" height="180">

FixRes is a simple method for fixing the train-test resolution discrepancy and thus improve the performance of any convolutional neural network.

The method is described in "Fixing the train-test resolution discrepancy" ([arXiv link](https://arxiv.org/abs/1906.06423)). 

BibTeX reference:
```bibtex
@ARTICLE{2019arXivFixRes,
       author = {Hugo, Touvron and Vedaldi, Andrea and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
        title = "{Fixing the train-test resolution discrepancy}",
      journal = {arXiv e-prints},
         year = "2019",
        month = "June",
}
```
Please cite it if you use it. 

# Kaggle contest

## Herbarium
### Results
Rank : 4 / 22

Score : 88.8 % top-1 accuracy

### Code
You have to separate the dataset in 3 parts train, validation and test.
Download ours [FixResNet-50](https://dl.fbaipublicfiles.com/FixRes_data/contest/Herbarium_data/herbarium_fixresnet50.pth) and [FixSENet154](https://dl.fbaipublicfiles.com/FixRes_data/contest/Herbarium_data/herbarium_fixsenet154.pth).
Download our [indice](https://dl.fbaipublicfiles.com/FixRes_data/contest/Herbarium_data/test_cp.csv) file.
Run the following script with your parameters :

```

python fixres_Herbarium.py --data-folder '/dataset/path' --num-workers 2 --batch-per-gpu 32 --indice-path '/indice/path' --output-path '' --senet-path '/senetpath/senet.pth' --resnet-path '/resnet/path/resnet.pth'

```


## iNaturalist 2019
### Results
Rank : 5 / 214

Score : 86.6 % top-1 accuracy

### Models

Download ours [FixSE-ResNext-101-32x4d](https://dl.fbaipublicfiles.com/FixRes_data/contest/INat_data/inat_fixseresnet32x4.pth), [FixSENet154](https://dl.fbaipublicfiles.com/FixRes_data/contest/INat_data/inat_fixsenet154.pth), [FixInception-ResNet-V2](https://dl.fbaipublicfiles.com/FixRes_data/contest/INat_data/inat_fixinception_resnetV2.pth) and [FixResNet-152-MPN-COV](https://dl.fbaipublicfiles.com/FixRes_data/contest/INat_data/inat_fixresnet_152_mpn.pth).
Download INaturalist 2019 dataset.
We have implement the ResNet-152 MPN-COV fron https://github.com/jiangtaoxie/MPN-COV. We don't provide the code of this model.
Run the following script with your parameters :
```
python fixres_INat.py

```

## Contributing
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
FixRes is [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) licensed, as found in the LICENSE file.

The Training from scratch implementation is based on https://github.com/facebookresearch/multigrain.

Model definition scripts are based on https://github.com/pytorch/vision/ and https://github.com/Cadene/pretrained-models.pytorch.
