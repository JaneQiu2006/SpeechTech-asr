# Deep-dive ASR error analysis

## System summary

| System | WER | S | D | I | Exact | Non-empty | Length ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| e13_layer10_bilstm_ctc | 0.2696 | 10547 | 684 | 725 | 45 | 1.0000 | 1.0009 |
| e13_layer10_bilstm_ctc | 0.2802 | 12989 | 764 | 979 | 56 | 1.0000 | 1.0041 |
| e13_layer11_bilstm_ctc | 0.4545 | 17371 | 1513 | 1274 | 4 | 1.0000 | 0.9946 |
| e13_layer11_bilstm_ctc | 0.4473 | 20271 | 1611 | 1635 | 7 | 1.0000 | 1.0005 |
| e13_layer1_bilstm_ctc | 0.6022 | 22543 | 2539 | 1625 | 4 | 1.0000 | 0.9794 |
| e13_layer1_bilstm_ctc | 0.5967 | 26485 | 2775 | 2114 | 3 | 1.0000 | 0.9874 |
| e13_layer2_bilstm_ctc | 0.5499 | 20579 | 2156 | 1654 | 1 | 1.0000 | 0.9887 |
| e13_layer2_bilstm_ctc | 0.5557 | 24615 | 2321 | 2278 | 5 | 1.0000 | 0.9992 |
| e13_layer3_bilstm_ctc | 0.5084 | 19029 | 1914 | 1604 | 8 | 1.0000 | 0.9930 |
| e13_layer3_bilstm_ctc | 0.5111 | 22657 | 2107 | 2105 | 7 | 1.0000 | 1.0000 |
| e13_layer4_bilstm_ctc | 0.4595 | 17208 | 1765 | 1405 | 13 | 1.0000 | 0.9919 |
| e13_layer4_bilstm_ctc | 0.4648 | 20658 | 1908 | 1873 | 13 | 1.0000 | 0.9993 |
| e13_layer5_bilstm_ctc | 0.3858 | 14552 | 1319 | 1237 | 25 | 1.0000 | 0.9982 |
| e13_layer5_bilstm_ctc | 0.3909 | 17445 | 1523 | 1583 | 19 | 1.0000 | 1.0011 |
| e13_layer7_bilstm_ctc | 0.2218 | 8628 | 588 | 619 | 154 | 1.0000 | 1.0007 |
| e13_layer7_bilstm_ctc | 0.2303 | 10601 | 665 | 842 | 165 | 1.0000 | 1.0034 |
| e13_layer8_bilstm_ctc | 0.1826 | 7195 | 445 | 459 | 241 | 1.0000 | 1.0003 |
| e13_layer8_bilstm_ctc | 0.1951 | 9058 | 501 | 701 | 226 | 1.0000 | 1.0038 |
| e14_layer_mixture_bilstm_ctc | 0.1958 | 7726 | 467 | 489 | 194 | 1.0000 | 1.0005 |
| e14_layer_mixture_bilstm_ctc | 0.2035 | 9511 | 543 | 643 | 220 | 1.0000 | 1.0019 |
| e15a_linear_layer8_ctc | 0.6449 | 24081 | 3931 | 590 | 0 | 0.9996 | 0.9247 |
| e15a_linear_layer8_ctc | 0.6512 | 28544 | 4987 | 709 | 0 | 0.9962 | 0.9186 |
| e15b_mlp_layer8_ctc | 0.3708 | 14107 | 1760 | 576 | 19 | 1.0000 | 0.9733 |
| e15b_mlp_layer8_ctc | 0.3815 | 17127 | 2240 | 690 | 28 | 0.9996 | 0.9705 |
| e15c_bilstm_layer8_ctc | 0.2351 | 9289 | 641 | 497 | 127 | 0.9996 | 0.9968 |
| e15c_bilstm_layer8_ctc | 0.2398 | 11258 | 702 | 650 | 125 | 1.0000 | 0.9990 |
| e15d_bilstm_layer8_ctc | 0.1830 | 7192 | 469 | 454 | 242 | 1.0000 | 0.9997 |
| e15d_bilstm_layer8_ctc | 0.1957 | 9073 | 530 | 688 | 225 | 1.0000 | 1.0030 |
| e16a_top4_finetune | 0.1387 | 5534 | 290 | 329 | 396 | 1.0000 | 1.0009 |
| e16a_top4_finetune | 0.1440 | 6834 | 321 | 417 | 394 | 1.0000 | 1.0018 |
| e16b_top5_finetune | 0.1218 | 4829 | 269 | 304 | 498 | 1.0000 | 1.0008 |
| e16b_top5_finetune | 0.1262 | 5941 | 286 | 407 | 504 | 1.0000 | 1.0023 |
| e16c_top8_finetune | 0.1160 | 4544 | 277 | 323 | 562 | 1.0000 | 1.0010 |
| e16c_top8_finetune | 0.1195 | 5559 | 324 | 401 | 584 | 1.0000 | 1.0015 |
| e17_k1000_centroid_bilstm_ctc | 0.3201 | 12435 | 976 | 786 | 43 | 1.0000 | 0.9957 |
| e17_k1000_centroid_bilstm_ctc | 0.3311 | 15280 | 1070 | 1058 | 45 | 1.0000 | 0.9998 |
| e17_k500_centroid_bilstm_ctc | 0.3677 | 14178 | 1202 | 927 | 24 | 1.0000 | 0.9938 |
| e17_k500_centroid_bilstm_ctc | 0.3788 | 17360 | 1332 | 1225 | 28 | 1.0000 | 0.9980 |
| e18_k200_embedding_bilstm_ctc | 0.4882 | 18501 | 1980 | 1168 | 7 | 1.0000 | 0.9817 |
| e18_k200_embedding_bilstm_ctc | 0.4985 | 22418 | 2146 | 1645 | 6 | 1.0000 | 0.9905 |
| e20a_1h_layer8_bilstm_ctc | 0.3063 | 12204 | 578 | 801 | 45 | 1.0000 | 1.0050 |
| e20a_1h_layer8_bilstm_ctc | 0.3163 | 14928 | 673 | 1029 | 45 | 1.0000 | 1.0068 |
| e20b_3h_layer8_bilstm_ctc | 0.2233 | 8812 | 557 | 533 | 146 | 1.0000 | 0.9995 |
| e20b_3h_layer8_bilstm_ctc | 0.2362 | 11022 | 583 | 816 | 152 | 1.0000 | 1.0044 |
| wav2vec2_discrete_k100_bilstm_ctc | 0.6126 | 23022 | 2565 | 1580 | 2 | 1.0000 | 0.9778 |
| wav2vec2_discrete_k100_bilstm_ctc | 0.6165 | 27504 | 2660 | 2247 | 2 | 1.0000 | 0.9921 |
| wav2vec2_discrete_k200_bilstm_ctc | 0.5231 | 19647 | 2074 | 1476 | 4 | 1.0000 | 0.9865 |
| wav2vec2_discrete_k200_bilstm_ctc | 0.5260 | 23435 | 2187 | 2032 | 5 | 1.0000 | 0.9971 |
| wav2vec2_discrete_k50_bilstm_ctc | 0.7122 | 27009 | 2940 | 1638 | 0 | 1.0000 | 0.9706 |
| wav2vec2_discrete_k50_bilstm_ctc | 0.7156 | 32103 | 3193 | 2325 | 0 | 1.0000 | 0.9835 |
| wav2vec2_finetune_10h | 0.1570 | 6376 | 295 | 293 | 1 | 1.0000 | 1.0000 |
| wav2vec2_finetune_1h | 1.0000 | 2509 | 41840 | 0 | 0 | 0.9984 | 0.0566 |
| wav2vec2_finetune_1h_repaired | 0.4017 | 16382 | 715 | 720 | 13 | 0.9996 | 1.0001 |
| wav2vec2_finetune_1h_time_mask | 0.4036 | 16633 | 578 | 687 | 14 | 1.0000 | 1.0025 |
| wav2vec2_finetune_1h_time_mask | 0.4073 | 19889 | 652 | 875 | 22 | 1.0000 | 1.0042 |
| wav2vec2_finetune_3h | 0.1669 | 6636 | 395 | 371 | 292 | 1.0000 | 0.9995 |
| wav2vec2_finetune_3h | 0.1712 | 8095 | 415 | 491 | 319 | 1.0000 | 1.0014 |
| wav2vec2_frozen_10h_fair_30ep | 0.9993 | 9319 | 35000 | 0 | 0 | 0.9180 | 0.2108 |
| wav2vec2_frozen_10h_v2 | 1.0000 | 0 | 50420 | 0 | 0 | 0.0000 | 0.0000 |
| wav2vec2_frozen_bilstm_10h | 1.0000 | 2513 | 41836 | 0 | 0 | 1.0000 | 0.0567 |
| wav2vec2_frozen_bilstm_10h | 1.0000 | 2620 | 49956 | 0 | 0 | 1.0000 | 0.0498 |
| wav2vec2_layer12_bilstm_ctc | 0.4727 | 17932 | 1653 | 1377 | 5 | 1.0000 | 0.9938 |
| wav2vec2_layer12_bilstm_ctc | 0.4618 | 20850 | 1766 | 1664 | 3 | 1.0000 | 0.9981 |
| wav2vec2_layer6_bilstm_ctc | 0.2901 | 11129 | 866 | 872 | 76 | 1.0000 | 1.0001 |
| wav2vec2_layer6_bilstm_ctc | 0.3003 | 13643 | 982 | 1164 | 72 | 1.0000 | 1.0035 |
| wav2vec2_layer9_bilstm_ctc | 0.1941 | 7687 | 423 | 500 | 171 | 1.0000 | 1.0017 |
| wav2vec2_layer9_bilstm_ctc | 0.2051 | 9608 | 477 | 698 | 183 | 1.0000 | 1.0042 |
| wav2vec2_masking_finetune_10h | 0.1098 | 4299 | 282 | 288 | 618 | 1.0000 | 1.0001 |
| wav2vec2_masking_finetune_10h | 0.1135 | 5268 | 311 | 390 | 608 | 1.0000 | 1.0015 |
| wav2vec2_top3_finetune_10h | 0.2195 | 8851 | 330 | 553 | 117 | 1.0000 | 1.0050 |
| wav2vec2_top3_finetune_10h | 0.2241 | 10780 | 375 | 629 | 130 | 1.0000 | 1.0048 |
| wav2vec2_top6_finetune_10h | 0.1192 | 4698 | 294 | 296 | 544 | 1.0000 | 1.0000 |
| wav2vec2_top6_finetune_10h | 0.1214 | 5714 | 301 | 368 | 545 | 1.0000 | 1.0013 |
| wavlm_finetune_10h | 0.1660 | 6440 | 488 | 434 | 322 | 1.0000 | 0.9988 |

## e13_layer10_bilstm_ctc

Top substitutions:

- `and` → `an`: 44
- `in` → `and`: 28
- `and` → `in`: 25
- `into` → `to`: 21
- `are` → `ar`: 18
- `too` → `to`: 16
- `in` → `an`: 16
- `his` → `is`: 15
- `their` → `ther`: 14
- `two` → `to`: 13
- `see` → `se`: 13
- `poor` → `por`: 12
- `too` → `two`: 12
- `had` → `ad`: 12
- `thee` → `the`: 12
- `know` → `now`: 11
- `what` → `wat`: 11
- `a` → `the`: 11
- `to` → `o`: 10
- `it's` → `its`: 10
- `been` → `ben`: 10
- `an` → `and`: 10
- `food` → `fod`: 10
- `and` → `ind`: 9
- `bartley` → `bartly`: 9
- `their` → `there`: 9
- `in` → `ind`: 9
- `is` → `his`: 9
- `either` → `ither`: 9
- `new` → `knew`: 8

Top deleted words:

- `a`: 70
- `in`: 22
- `the`: 13
- `to`: 12
- `i`: 11
- `and`: 10
- `with`: 7
- `is`: 7
- `this`: 6
- `some`: 6
- `are`: 6
- `can`: 6
- `he`: 6
- `what`: 6
- `will`: 5
- `they`: 5
- `there`: 5
- `how`: 4
- `do`: 4
- `could`: 4
- `old`: 4
- `new`: 4
- `were`: 4
- `but`: 4
- `had`: 4
- `any`: 4
- `our`: 4
- `his`: 4
- `left`: 4
- `shall`: 4

Top inserted words:

- `a`: 53
- `in`: 38
- `i`: 12
- `an`: 11
- `the`: 11
- `for`: 11
- `over`: 11
- `any`: 8
- `with`: 7
- `some`: 7
- `and`: 6
- `out`: 6
- `be`: 6
- `to`: 6
- `un`: 6
- `all`: 5
- `every`: 5
- `at`: 5
- `there`: 5
- `al`: 5
- `no`: 5
- `under`: 4
- `he`: 4
- `it`: 4
- `de`: 4
- `of`: 4
- `news`: 4
- `im`: 4
- `mage`: 4
- `wen`: 3

Duration-bucket WER:

- 5-10s: 0.2410
- <5s: 0.3822
- 10-15s: 0.2254

Highest speaker-level WER:

- 777: 0.3529
- 1272: 0.3353
- 652: 0.3344
- 5694: 0.3194
- 1988: 0.3063
- 6313: 0.3054
- 3752: 0.3041
- 251: 0.3003
- 1919: 0.2916
- 5895: 0.2896
- 2803: 0.2822
- 2902: 0.2810
- 1673: 0.2804
- 5338: 0.2798
- 2412: 0.2747
- 1462: 0.2722
- 2078: 0.2704
- 6241: 0.2697
- 7850: 0.2689
- 2035: 0.2664
- 8842: 0.2656
- 84: 0.2633
- 5536: 0.2572
- 2086: 0.2564
- 6295: 0.2547
- 3000: 0.2535
- 2428: 0.2524
- 3081: 0.2519
- 6319: 0.2508
- 174: 0.2500

Representative high-error utterances:

- `1919-142785-0039` ref: illustration marjoram / hyp: i le shration miurom
- `2078-142845-0009` ref: illustration italian millet / hyp: il strition ait t ant melte
- `8842-304647-0001` ref: quinci impara a stupirti / hyp: qunt se im pd as to bearty
- `1919-142785-0000` ref: illustration long pepper / hyp: i l sration mon peper
- `3081-166546-0021` ref: i emphasised complacently / hyp: om the sised compli scently

## e13_layer10_bilstm_ctc

Top substitutions:

- `and` → `in`: 45
- `and` → `an`: 41
- `thee` → `the`: 26
- `are` → `ar`: 24
- `is` → `his`: 22
- `in` → `and`: 21
- `in` → `an`: 21
- `too` → `to`: 20
- `their` → `ther`: 20
- `been` → `ben`: 17
- `an` → `and`: 17
- `it's` → `its`: 16
- `too` → `two`: 15
- `oh` → `o`: 15
- `heart` → `hart`: 14
- `into` → `to`: 12
- `is` → `as`: 12
- `know` → `now`: 12
- `and` → `ind`: 12
- `a` → `the`: 12
- `paul` → `pall`: 12
- `christ` → `crised`: 11
- `this` → `the`: 11
- `there` → `ther`: 11
- `in` → `ind`: 11
- `where` → `were`: 11
- `know` → `no`: 11
- `two` → `to`: 10
- `his` → `is`: 10
- `our` → `ar`: 10

Top deleted words:

- `a`: 80
- `i`: 20
- `in`: 16
- `be`: 15
- `it`: 10
- `you`: 10
- `the`: 10
- `to`: 10
- `is`: 9
- `and`: 8
- `fir`: 8
- `all`: 7
- `he`: 6
- `there`: 6
- `free`: 6
- `this`: 5
- `more`: 5
- `no`: 5
- `some`: 5
- `for`: 5
- `do`: 5
- `we`: 5
- `am`: 5
- `an`: 5
- `know`: 4
- `that`: 4
- `miss`: 4
- `with`: 4
- `her`: 4
- `latter`: 4

Top inserted words:

- `a`: 70
- `in`: 43
- `some`: 14
- `an`: 14
- `to`: 14
- `with`: 13
- `i`: 11
- `the`: 11
- `any`: 10
- `un`: 10
- `be`: 9
- `and`: 9
- `o`: 9
- `over`: 8
- `what`: 7
- `all`: 7
- `e`: 7
- `every`: 7
- `out`: 7
- `ad`: 6
- `never`: 6
- `is`: 6
- `he`: 6
- `for`: 6
- `on`: 5
- `house`: 5
- `inter`: 5
- `capt`: 5
- `there`: 4
- `her`: 4

Duration-bucket WER:

- 10-15s: 0.2356
- <5s: 0.4044
- 5-10s: 0.2719
- >=15s: 0.2341

Highest speaker-level WER:

- 3570: 0.3815
- 1995: 0.3615
- 8555: 0.3551
- 7729: 0.3362
- 8463: 0.3355
- 2961: 0.3322
- 4446: 0.3294
- 4507: 0.3250
- 7176: 0.3222
- 5142: 0.3156
- 4992: 0.3143
- 3575: 0.3123
- 260: 0.3044
- 8455: 0.3023
- 2830: 0.2987
- 4077: 0.2978
- 6829: 0.2951
- 2094: 0.2938
- 8224: 0.2903
- 2300: 0.2881
- 4970: 0.2787
- 908: 0.2626
- 1188: 0.2616
- 61: 0.2593
- 5683: 0.2576
- 121: 0.2544
- 237: 0.2540
- 5105: 0.2533
- 7127: 0.2506
- 1284: 0.2492

Representative high-error utterances:

- `3575-170457-0016` ref: farewell madam / hyp: far bo the tdn
- `1995-1836-0011` ref: positively heroic added cresswell avoiding his sister's eyes / hyp: bosid ie e nor alict aind i cors wall a orting e istis is
- `5105-28241-0010` ref: ocean reigned supreme / hyp: o shan raint sor prame
- `6930-81414-0002` ref: onward said a distant voice / hyp: be he wer saidt it dest in bice
- `1089-134691-0003` ref: the university / hyp: the yunt ofe vrsity

## e13_layer11_bilstm_ctc

Top substitutions:

- `and` → `an`: 61
- `and` → `in`: 61
- `in` → `and`: 47
- `his` → `is`: 43
- `in` → `an`: 43
- `is` → `as`: 32
- `two` → `to`: 29
- `they` → `the`: 28
- `all` → `al`: 26
- `too` → `to`: 26
- `in` → `ind`: 25
- `their` → `ther`: 25
- `it` → `at`: 24
- `are` → `ar`: 23
- `as` → `is`: 22
- `into` → `to`: 20
- `see` → `se`: 20
- `been` → `ben`: 19
- `had` → `ad`: 19
- `four` → `for`: 19
- `is` → `his`: 18
- `the` → `he`: 17
- `said` → `sad`: 17
- `and` → `ind`: 16
- `of` → `o`: 16
- `there` → `ther`: 15
- `it` → `i`: 15
- `now` → `no`: 14
- `with` → `wit`: 14
- `when` → `wen`: 14

Top deleted words:

- `a`: 152
- `i`: 32
- `to`: 31
- `he`: 25
- `the`: 24
- `in`: 23
- `and`: 21
- `can`: 16
- `we`: 13
- `for`: 13
- `be`: 12
- `how`: 11
- `you`: 11
- `an`: 11
- `were`: 10
- `it`: 10
- `there`: 10
- `what`: 10
- `they`: 9
- `up`: 9
- `of`: 8
- `some`: 8
- `me`: 8
- `are`: 8
- `is`: 7
- `see`: 7
- `her`: 7
- `all`: 7
- `few`: 7
- `as`: 7

Top inserted words:

- `a`: 81
- `in`: 36
- `an`: 19
- `the`: 16
- `to`: 15
- `and`: 13
- `i`: 12
- `at`: 12
- `be`: 11
- `for`: 11
- `over`: 10
- `with`: 10
- `e`: 10
- `un`: 10
- `o`: 9
- `some`: 8
- `no`: 7
- `ar`: 7
- `all`: 7
- `s`: 7
- `on`: 7
- `al`: 7
- `any`: 7
- `out`: 6
- `wen`: 6
- `as`: 6
- `im`: 6
- `or`: 6
- `te`: 6
- `there`: 6

Duration-bucket WER:

- 5-10s: 0.4312
- <5s: 0.5624
- 10-15s: 0.4053

Highest speaker-level WER:

- 777: 0.5567
- 6313: 0.5476
- 6241: 0.5181
- 1988: 0.5137
- 5694: 0.5130
- 1272: 0.5015
- 5338: 0.5000
- 652: 0.4937
- 5536: 0.4913
- 3752: 0.4882
- 2412: 0.4747
- 8842: 0.4671
- 1919: 0.4641
- 1462: 0.4579
- 251: 0.4566
- 1673: 0.4554
- 422: 0.4529
- 5895: 0.4517
- 2803: 0.4470
- 7850: 0.4469
- 7976: 0.4424
- 2086: 0.4423
- 2035: 0.4422
- 174: 0.4363
- 2078: 0.4320
- 3170: 0.4303
- 3081: 0.4297
- 2277: 0.4283
- 84: 0.4232
- 2428: 0.4230

Representative high-error utterances:

- `3081-166546-0005` ref: george nodded / hyp: gor o not itd
- `1462-170138-0003` ref: alexander exclaimed mildly / hyp: allesen eur exlaime to mioldly
- `174-50561-0009` ref: the wandering singer / hyp: fo o d ry saner
- `2078-142845-0000` ref: kirkleatham yeast / hyp: curcly thom yast
- `2078-142845-0001` ref: seventeen seventeen / hyp: seventen seven ten

## e13_layer11_bilstm_ctc

Top substitutions:

- `and` → `in`: 77
- `and` → `an`: 77
- `in` → `and`: 66
- `in` → `an`: 50
- `is` → `as`: 33
- `are` → `ar`: 31
- `an` → `and`: 30
- `in` → `ind`: 30
- `their` → `ther`: 27
- `is` → `his`: 26
- `all` → `al`: 25
- `too` → `to`: 25
- `two` → `to`: 24
- `it` → `at`: 24
- `into` → `to`: 23
- `thee` → `the`: 23
- `at` → `it`: 21
- `will` → `wil`: 20
- `there` → `ther`: 19
- `know` → `no`: 19
- `than` → `then`: 18
- `the` → `te`: 18
- `said` → `sad`: 17
- `too` → `two`: 17
- `own` → `on`: 16
- `still` → `stil`: 16
- `of` → `o`: 16
- `they` → `the`: 15
- `you` → `yu`: 15
- `and` → `ind`: 15

Top deleted words:

- `a`: 154
- `i`: 43
- `in`: 36
- `the`: 28
- `to`: 27
- `it`: 26
- `you`: 24
- `be`: 21
- `and`: 19
- `can`: 13
- `all`: 13
- `will`: 12
- `of`: 12
- `is`: 11
- `there`: 11
- `with`: 11
- `for`: 11
- `her`: 11
- `are`: 10
- `an`: 10
- `they`: 10
- `or`: 9
- `so`: 9
- `some`: 9
- `he`: 9
- `were`: 8
- `at`: 8
- `our`: 8
- `we`: 8
- `but`: 7

Top inserted words:

- `a`: 126
- `in`: 42
- `and`: 26
- `an`: 22
- `the`: 21
- `e`: 18
- `o`: 18
- `to`: 18
- `be`: 16
- `on`: 14
- `i`: 14
- `un`: 12
- `as`: 11
- `any`: 11
- `some`: 11
- `for`: 11
- `ar`: 10
- `all`: 10
- `al`: 10
- `with`: 9
- `ther`: 8
- `is`: 8
- `her`: 8
- `at`: 8
- `over`: 8
- `af`: 7
- `te`: 7
- `out`: 7
- `de`: 7
- `no`: 6

Duration-bucket WER:

- 10-15s: 0.4051
- <5s: 0.5691
- 5-10s: 0.4437
- >=15s: 0.3934

Highest speaker-level WER:

- 3570: 0.5530
- 4446: 0.5275
- 5142: 0.5263
- 1995: 0.5188
- 260: 0.5086
- 2830: 0.5022
- 8463: 0.4992
- 7176: 0.4942
- 8455: 0.4920
- 4507: 0.4854
- 8555: 0.4844
- 4077: 0.4838
- 4992: 0.4829
- 7729: 0.4770
- 3575: 0.4692
- 2961: 0.4671
- 2094: 0.4579
- 61: 0.4578
- 237: 0.4568
- 6829: 0.4517
- 1188: 0.4506
- 2300: 0.4501
- 4970: 0.4337
- 8224: 0.4291
- 6930: 0.4288
- 1580: 0.4236
- 121: 0.4235
- 7127: 0.4166
- 5683: 0.4147
- 5105: 0.4052

Representative high-error utterances:

- `260-123286-0010` ref: sunday august sixteenth / hyp: son day a go sixtate
- `260-123286-0020` ref: tuesday august eighteenth / hyp: tuse day ol tist ighteethi
- `260-123288-0014` ref: hans stirs not / hyp: fhor e s fured norit
- `1995-1836-0011` ref: positively heroic added cresswell avoiding his sister's eyes / hyp: os i tei mey hear alik anet cres weall a voadtn his sistures ie
- `1089-134691-0003` ref: the university / hyp: the un of varsity

## e13_layer1_bilstm_ctc

Top substitutions:

- `and` → `an`: 85
- `and` → `in`: 59
- `in` → `an`: 48
- `in` → `and`: 46
- `his` → `is`: 40
- `their` → `ther`: 35
- `into` → `to`: 30
- `is` → `his`: 29
- `all` → `al`: 27
- `of` → `a`: 26
- `of` → `o`: 24
- `the` → `he`: 23
- `it` → `at`: 22
- `is` → `as`: 22
- `the` → `a`: 22
- `they` → `the`: 21
- `two` → `to`: 21
- `the` → `te`: 21
- `the` → `tha`: 21
- `too` → `to`: 20
- `as` → `is`: 19
- `that` → `the`: 18
- `a` → `the`: 17
- `at` → `it`: 17
- `there` → `ther`: 17
- `see` → `se`: 16
- `a` → `o`: 16
- `and` → `ind`: 16
- `been` → `ben`: 16
- `in` → `ind`: 15

Top deleted words:

- `a`: 146
- `the`: 72
- `to`: 50
- `in`: 48
- `and`: 43
- `i`: 40
- `it`: 38
- `he`: 34
- `of`: 25
- `you`: 24
- `we`: 19
- `as`: 19
- `for`: 18
- `have`: 18
- `her`: 18
- `how`: 17
- `is`: 15
- `they`: 15
- `that`: 15
- `see`: 15
- `are`: 15
- `had`: 14
- `if`: 14
- `do`: 14
- `this`: 13
- `was`: 13
- `with`: 13
- `be`: 12
- `were`: 12
- `time`: 12

Top inserted words:

- `a`: 68
- `in`: 40
- `and`: 20
- `o`: 17
- `an`: 17
- `to`: 16
- `the`: 16
- `he`: 16
- `i`: 14
- `with`: 11
- `e`: 11
- `is`: 10
- `s`: 10
- `of`: 10
- `te`: 9
- `it`: 9
- `be`: 9
- `for`: 9
- `on`: 8
- `ad`: 8
- `al`: 7
- `over`: 7
- `at`: 7
- `man`: 7
- `as`: 7
- `im`: 6
- `not`: 6
- `ind`: 5
- `all`: 5
- `or`: 5

Duration-bucket WER:

- 5-10s: 0.5851
- <5s: 0.6543
- 10-15s: 0.5882

Highest speaker-level WER:

- 1272: 0.7423
- 6313: 0.7247
- 5536: 0.6857
- 2803: 0.6828
- 5338: 0.6781
- 777: 0.6693
- 5694: 0.6466
- 1988: 0.6463
- 7850: 0.6322
- 8842: 0.6298
- 2412: 0.6287
- 174: 0.6262
- 6241: 0.6182
- 422: 0.6176
- 2902: 0.6161
- 652: 0.6136
- 1673: 0.6095
- 3081: 0.6083
- 3853: 0.6069
- 2035: 0.5991
- 3576: 0.5982
- 6345: 0.5833
- 2086: 0.5806
- 251: 0.5767
- 84: 0.5752
- 3752: 0.5727
- 7976: 0.5725
- 6295: 0.5667
- 5895: 0.5651
- 2277: 0.5607

Representative high-error utterances:

- `3081-166546-0021` ref: i emphasised complacently / hyp: y ampt the cized to complayesosleae o
- `3081-166546-0005` ref: george nodded / hyp: e wajortgi nod eide
- `5694-64038-0000` ref: advance into tennessee / hyp: ad vance an detene a s
- `1462-170145-0021` ref: alexander flushed angrily / hyp: ailis end der flushet angarlly
- `1272-135031-0019` ref: that's funny remarked betsy thoughtfully / hyp: that's funna r market a but see thought fouly

## e13_layer1_bilstm_ctc

Top substitutions:

- `and` → `an`: 93
- `and` → `in`: 81
- `in` → `and`: 72
- `in` → `an`: 61
- `is` → `as`: 33
- `the` → `he`: 33
- `in` → `ind`: 29
- `his` → `is`: 28
- `it` → `at`: 26
- `an` → `and`: 24
- `the` → `te`: 24
- `into` → `to`: 23
- `that` → `the`: 23
- `their` → `ther`: 23
- `too` → `to`: 22
- `all` → `al`: 22
- `is` → `his`: 22
- `of` → `af`: 22
- `a` → `o`: 21
- `of` → `o`: 21
- `are` → `ar`: 21
- `two` → `to`: 20
- `a` → `the`: 20
- `as` → `is`: 20
- `of` → `a`: 19
- `the` → `tha`: 19
- `thee` → `the`: 17
- `own` → `on`: 16
- `will` → `wil`: 16
- `there` → `ther`: 16

Top deleted words:

- `a`: 166
- `the`: 67
- `and`: 57
- `i`: 53
- `in`: 49
- `to`: 48
- `of`: 32
- `you`: 28
- `for`: 28
- `he`: 27
- `be`: 27
- `her`: 24
- `but`: 20
- `not`: 20
- `have`: 19
- `it`: 19
- `is`: 19
- `had`: 18
- `will`: 17
- `at`: 17
- `we`: 16
- `am`: 16
- `who`: 16
- `or`: 15
- `are`: 15
- `as`: 15
- `all`: 14
- `they`: 14
- `can`: 13
- `was`: 13

Top inserted words:

- `a`: 89
- `in`: 51
- `an`: 32
- `o`: 28
- `and`: 28
- `the`: 23
- `i`: 21
- `to`: 19
- `he`: 13
- `as`: 13
- `at`: 12
- `some`: 12
- `ad`: 12
- `be`: 11
- `for`: 11
- `af`: 10
- `e`: 10
- `with`: 10
- `of`: 10
- `ha`: 9
- `is`: 9
- `ther`: 9
- `it`: 9
- `there`: 8
- `on`: 8
- `un`: 8
- `man`: 8
- `all`: 7
- `has`: 7
- `s`: 7

Duration-bucket WER:

- 10-15s: 0.5822
- <5s: 0.6469
- 5-10s: 0.5907
- >=15s: 0.5785

Highest speaker-level WER:

- 8555: 0.7036
- 4507: 0.6802
- 3570: 0.6766
- 8463: 0.6710
- 4992: 0.6627
- 5142: 0.6557
- 7729: 0.6533
- 4446: 0.6392
- 8455: 0.6352
- 1995: 0.6346
- 4970: 0.6345
- 2961: 0.6341
- 2300: 0.6334
- 4077: 0.6227
- 7176: 0.6225
- 1188: 0.6211
- 6930: 0.6192
- 237: 0.6180
- 260: 0.6150
- 121: 0.6050
- 8224: 0.6012
- 2094: 0.6003
- 6829: 0.6003
- 3575: 0.5996
- 5683: 0.5974
- 2830: 0.5960
- 61: 0.5949
- 5639: 0.5894
- 3729: 0.5784
- 908: 0.5682

Representative high-error utterances:

- `1995-1836-0011` ref: positively heroic added cresswell avoiding his sister's eyes / hyp: pos i toev ley heo oic addi curus well a vorning he sistours eys
- `4507-16021-0040` ref: one thinks one hears hydras talking / hyp: on tae s won shear as hig tras s talki
- `1089-134691-0003` ref: the university / hyp: the youn av virsity
- `1089-134691-0018` ref: again again / hyp: a gan agin
- `1089-134691-0024` ref: stephanos dedalos / hyp: stuff inost dedos

## e13_layer2_bilstm_ctc

Top substitutions:

- `and` → `an`: 82
- `in` → `an`: 54
- `and` → `in`: 52
- `his` → `is`: 44
- `in` → `and`: 42
- `into` → `to`: 33
- `it` → `at`: 30
- `there` → `ther`: 24
- `are` → `ar`: 23
- `is` → `as`: 21
- `of` → `a`: 21
- `two` → `to`: 20
- `as` → `is`: 20
- `all` → `al`: 19
- `their` → `ther`: 19
- `that` → `the`: 18
- `too` → `to`: 18
- `a` → `the`: 18
- `of` → `o`: 17
- `to` → `o`: 17
- `had` → `ad`: 17
- `the` → `te`: 16
- `is` → `his`: 16
- `of` → `af`: 16
- `at` → `it`: 16
- `been` → `ben`: 16
- `that` → `at`: 16
- `the` → `tha`: 15
- `see` → `se`: 15
- `this` → `the`: 14

Top deleted words:

- `a`: 126
- `and`: 62
- `the`: 53
- `to`: 43
- `i`: 41
- `in`: 40
- `he`: 30
- `of`: 27
- `you`: 24
- `all`: 22
- `it`: 20
- `we`: 18
- `be`: 15
- `are`: 15
- `is`: 14
- `with`: 13
- `were`: 12
- `had`: 12
- `on`: 12
- `can`: 12
- `this`: 11
- `his`: 11
- `they`: 11
- `how`: 11
- `well`: 11
- `an`: 11
- `as`: 10
- `not`: 10
- `out`: 10
- `first`: 10

Top inserted words:

- `a`: 67
- `in`: 49
- `an`: 31
- `to`: 30
- `the`: 21
- `o`: 16
- `and`: 15
- `i`: 14
- `he`: 13
- `ad`: 11
- `of`: 11
- `be`: 10
- `for`: 10
- `is`: 8
- `with`: 8
- `im`: 8
- `e`: 8
- `as`: 8
- `s`: 8
- `at`: 7
- `over`: 7
- `ap`: 6
- `hand`: 6
- `do`: 6
- `al`: 6
- `ind`: 6
- `t`: 5
- `af`: 5
- `so`: 5
- `it`: 5

Duration-bucket WER:

- 5-10s: 0.5304
- <5s: 0.6096
- 10-15s: 0.5339

Highest speaker-level WER:

- 1272: 0.6975
- 6313: 0.6884
- 777: 0.6377
- 2803: 0.6253
- 5338: 0.6194
- 5536: 0.6112
- 5694: 0.5921
- 2902: 0.5864
- 174: 0.5850
- 1988: 0.5844
- 8842: 0.5830
- 6241: 0.5763
- 422: 0.5627
- 2412: 0.5608
- 3853: 0.5576
- 7850: 0.5540
- 652: 0.5521
- 1919: 0.5513
- 1673: 0.5504
- 3081: 0.5462
- 2086: 0.5434
- 251: 0.5424
- 3752: 0.5387
- 6345: 0.5309
- 3000: 0.5284
- 7976: 0.5264
- 2078: 0.5200
- 8297: 0.5153
- 3170: 0.5151
- 2035: 0.5148

Representative high-error utterances:

- `3081-166546-0021` ref: i emphasised complacently / hyp: i am tho cise to com play somsly
- `2078-142845-0003` ref: seventeen eighteen / hyp: seven to eneight tdane
- `8842-304647-0001` ref: quinci impara a stupirti / hyp: qwueen tho im pat ous to bearty
- `1462-170138-0003` ref: alexander exclaimed mildly / hyp: allas ender exclame to miledly
- `2078-142845-0009` ref: illustration italian millet / hyp: illstration at taiy an melit

## e13_layer2_bilstm_ctc

Top substitutions:

- `and` → `an`: 108
- `and` → `in`: 78
- `in` → `and`: 57
- `in` → `an`: 47
- `in` → `ind`: 34
- `into` → `to`: 31
- `his` → `is`: 30
- `is` → `as`: 29
- `are` → `ar`: 28
- `of` → `a`: 27
- `all` → `al`: 27
- `the` → `tha`: 25
- `it` → `at`: 25
- `an` → `and`: 24
- `their` → `ther`: 24
- `of` → `o`: 23
- `too` → `to`: 22
- `as` → `is`: 22
- `been` → `ben`: 21
- `a` → `o`: 21
- `the` → `he`: 21
- `there` → `ther`: 20
- `two` → `to`: 19
- `it` → `i`: 19
- `is` → `his`: 18
- `that` → `the`: 18
- `with` → `wit`: 18
- `a` → `i`: 18
- `room` → `rom`: 17
- `and` → `ind`: 16

Top deleted words:

- `a`: 171
- `the`: 61
- `to`: 43
- `i`: 42
- `and`: 41
- `in`: 38
- `you`: 34
- `of`: 34
- `he`: 27
- `it`: 18
- `some`: 18
- `for`: 18
- `be`: 18
- `but`: 17
- `so`: 17
- `is`: 16
- `her`: 16
- `at`: 14
- `am`: 13
- `can`: 12
- `there`: 12
- `who`: 12
- `or`: 11
- `all`: 11
- `have`: 11
- `on`: 11
- `do`: 11
- `that`: 11
- `not`: 10
- `are`: 10

Top inserted words:

- `a`: 111
- `in`: 57
- `and`: 34
- `an`: 30
- `i`: 29
- `to`: 27
- `the`: 26
- `o`: 21
- `be`: 21
- `as`: 20
- `it`: 13
- `some`: 12
- `de`: 12
- `ad`: 12
- `ther`: 12
- `with`: 11
- `on`: 11
- `at`: 10
- `or`: 10
- `re`: 10
- `e`: 10
- `con`: 9
- `he`: 8
- `un`: 8
- `ha`: 8
- `t`: 8
- `s`: 8
- `of`: 8
- `fe`: 8
- `no`: 8

Duration-bucket WER:

- 10-15s: 0.5399
- <5s: 0.6074
- 5-10s: 0.5528
- >=15s: 0.5326

Highest speaker-level WER:

- 8463: 0.6524
- 8555: 0.6360
- 3570: 0.6313
- 1995: 0.6275
- 4992: 0.6196
- 4507: 0.6188
- 2961: 0.6176
- 7176: 0.6068
- 5142: 0.6060
- 4077: 0.6042
- 4446: 0.6007
- 8455: 0.5996
- 7729: 0.5986
- 260: 0.5798
- 3575: 0.5731
- 61: 0.5726
- 237: 0.5719
- 8224: 0.5709
- 2300: 0.5704
- 1188: 0.5656
- 5683: 0.5644
- 6829: 0.5577
- 2094: 0.5570
- 2830: 0.5534
- 4970: 0.5528
- 121: 0.5516
- 6930: 0.5480
- 5639: 0.5368
- 7127: 0.5217
- 1284: 0.5111

Representative high-error utterances:

- `1089-134691-0003` ref: the university / hyp: they oun o vercity
- `1995-1836-0011` ref: positively heroic added cresswell avoiding his sister's eyes / hyp: baus a dve te o alick adt cres well a voding he cisters aeyes
- `5105-28241-0014` ref: another circumstance was most remarkable / hyp: an ho this sacon stands was umust r markable
- `1089-134691-0018` ref: again again / hyp: agan ha gan
- `1089-134691-0024` ref: stephanos dedalos / hyp: stefin ao stediles

## e13_layer3_bilstm_ctc

Top substitutions:

- `and` → `in`: 72
- `and` → `an`: 70
- `in` → `and`: 47
- `in` → `an`: 42
- `his` → `is`: 34
- `it` → `at`: 32
- `into` → `to`: 30
- `two` → `to`: 26
- `is` → `his`: 22
- `are` → `ar`: 21
- `and` → `ind`: 21
- `too` → `to`: 21
- `their` → `ther`: 21
- `at` → `it`: 20
- `they` → `the`: 18
- `of` → `o`: 18
- `is` → `as`: 17
- `as` → `is`: 17
- `to` → `o`: 17
- `of` → `a`: 17
- `where` → `were`: 15
- `the` → `a`: 15
- `will` → `wil`: 15
- `a` → `the`: 15
- `to` → `a`: 15
- `there` → `ther`: 14
- `a` → `i`: 14
- `in` → `ind`: 13
- `an` → `in`: 13
- `i` → `a`: 13

Top deleted words:

- `a`: 134
- `the`: 46
- `to`: 41
- `and`: 37
- `i`: 36
- `in`: 33
- `of`: 23
- `it`: 20
- `we`: 18
- `all`: 16
- `had`: 15
- `you`: 14
- `for`: 14
- `on`: 14
- `he`: 13
- `with`: 13
- `are`: 12
- `can`: 11
- `see`: 11
- `were`: 10
- `do`: 10
- `be`: 10
- `as`: 10
- `how`: 10
- `that`: 9
- `some`: 9
- `what`: 9
- `his`: 8
- `not`: 8
- `been`: 8

Top inserted words:

- `a`: 72
- `in`: 50
- `an`: 29
- `and`: 26
- `to`: 21
- `the`: 20
- `o`: 18
- `i`: 17
- `he`: 14
- `at`: 13
- `of`: 12
- `with`: 11
- `im`: 11
- `be`: 11
- `for`: 10
- `e`: 10
- `s`: 9
- `is`: 9
- `al`: 8
- `ar`: 8
- `as`: 8
- `over`: 8
- `all`: 7
- `un`: 7
- `so`: 7
- `t`: 7
- `re`: 7
- `ther`: 7
- `some`: 6
- `on`: 6

Duration-bucket WER:

- 5-10s: 0.4894
- <5s: 0.5775
- 10-15s: 0.4836

Highest speaker-level WER:

- 1272: 0.6418
- 6313: 0.6239
- 777: 0.6126
- 5694: 0.5739
- 2803: 0.5722
- 1988: 0.5643
- 5536: 0.5575
- 5338: 0.5558
- 174: 0.5550
- 8842: 0.5502
- 2412: 0.5404
- 2902: 0.5393
- 422: 0.5392
- 3853: 0.5385
- 6241: 0.5295
- 7850: 0.5252
- 652: 0.5221
- 1673: 0.5203
- 3752: 0.5150
- 1919: 0.4938
- 5895: 0.4916
- 7976: 0.4914
- 2086: 0.4871
- 3170: 0.4816
- 2035: 0.4807
- 6345: 0.4805
- 251: 0.4757
- 2078: 0.4752
- 6295: 0.4726
- 3000: 0.4704

Representative high-error utterances:

- `8297-275155-0021` ref: honestly / hyp: o nestly
- `8842-304647-0001` ref: quinci impara a stupirti / hyp: qwun she im pad ase to beaict
- `1272-135031-0022` ref: true agreed kaliko / hyp: to o and gred calico
- `1462-170138-0003` ref: alexander exclaimed mildly / hyp: all a ender exclimt mioudly
- `1462-170145-0021` ref: alexander flushed angrily / hyp: all a ender flusht angrally

## e13_layer3_bilstm_ctc

Top substitutions:

- `and` → `in`: 88
- `and` → `an`: 83
- `in` → `and`: 63
- `in` → `an`: 39
- `are` → `ar`: 35
- `an` → `and`: 33
- `his` → `is`: 31
- `of` → `o`: 30
- `is` → `his`: 28
- `is` → `as`: 28
- `of` → `a`: 27
- `the` → `he`: 27
- `into` → `to`: 24
- `too` → `to`: 24
- `it` → `at`: 23
- `thee` → `the`: 21
- `too` → `two`: 21
- `as` → `is`: 21
- `the` → `te`: 21
- `at` → `it`: 20
- `all` → `al`: 20
- `know` → `no`: 20
- `two` → `to`: 19
- `to` → `o`: 18
- `their` → `ther`: 17
- `in` → `ind`: 17
- `that` → `the`: 17
- `the` → `a`: 17
- `there` → `ther`: 17
- `where` → `were`: 16

Top deleted words:

- `a`: 182
- `to`: 48
- `the`: 40
- `and`: 39
- `i`: 37
- `in`: 35
- `you`: 26
- `of`: 21
- `it`: 21
- `for`: 21
- `her`: 17
- `be`: 17
- `had`: 15
- `is`: 13
- `are`: 13
- `have`: 13
- `at`: 13
- `this`: 12
- `all`: 12
- `he`: 12
- `but`: 12
- `well`: 11
- `am`: 11
- `can`: 10
- `there`: 10
- `we`: 10
- `our`: 10
- `an`: 10
- `not`: 10
- `who`: 10

Top inserted words:

- `a`: 121
- `in`: 61
- `the`: 24
- `an`: 23
- `and`: 23
- `to`: 19
- `i`: 19
- `he`: 18
- `be`: 16
- `some`: 15
- `o`: 15
- `ad`: 11
- `as`: 11
- `is`: 10
- `s`: 10
- `te`: 10
- `it`: 10
- `re`: 10
- `all`: 9
- `al`: 9
- `on`: 8
- `out`: 8
- `ar`: 8
- `im`: 8
- `ap`: 7
- `of`: 7
- `no`: 7
- `ther`: 7
- `her`: 7
- `you`: 7

Duration-bucket WER:

- 10-15s: 0.4892
- <5s: 0.5783
- 5-10s: 0.5095
- >=15s: 0.4792

Highest speaker-level WER:

- 3570: 0.6070
- 1995: 0.5939
- 8555: 0.5892
- 8463: 0.5847
- 4992: 0.5676
- 4077: 0.5640
- 7176: 0.5631
- 5142: 0.5611
- 4507: 0.5604
- 2961: 0.5597
- 7729: 0.5552
- 8455: 0.5501
- 6829: 0.5456
- 4446: 0.5431
- 3575: 0.5408
- 2300: 0.5401
- 4970: 0.5371
- 260: 0.5274
- 237: 0.5216
- 1188: 0.5162
- 2830: 0.5152
- 8224: 0.5093
- 61: 0.5037
- 5683: 0.5032
- 6930: 0.4954
- 5639: 0.4877
- 121: 0.4867
- 7127: 0.4798
- 2094: 0.4787
- 908: 0.4730

Representative high-error utterances:

- `1089-134691-0024` ref: stephanos dedalos / hyp: staf inos ded nos
- `3575-170457-0016` ref: farewell madam / hyp: feare bo ed hem
- `8555-292519-0002` ref: venice / hyp: wt a
- `5105-28241-0014` ref: another circumstance was most remarkable / hyp: and ho the saicon s ant was mos fr morcable
- `1995-1836-0011` ref: positively heroic added cresswell avoiding his sister's eyes / hyp: bas i te nee ar allick adt at crst well alorting hescistrse ize

## e13_layer4_bilstm_ctc

Top substitutions:

- `and` → `an`: 57
- `in` → `and`: 54
- `in` → `an`: 52
- `and` → `in`: 50
- `his` → `is`: 42
- `it` → `at`: 34
- `into` → `to`: 29
- `are` → `ar`: 25
- `is` → `as`: 25
- `two` → `to`: 21
- `been` → `ben`: 21
- `of` → `a`: 21
- `there` → `ther`: 20
- `in` → `ind`: 19
- `as` → `is`: 18
- `to` → `a`: 18
- `at` → `it`: 17
- `their` → `ther`: 17
- `the` → `a`: 16
- `where` → `were`: 16
- `a` → `o`: 16
- `one` → `on`: 15
- `too` → `to`: 15
- `of` → `o`: 15
- `they` → `the`: 14
- `a` → `the`: 14
- `to` → `o`: 14
- `too` → `two`: 13
- `then` → `than`: 13
- `is` → `his`: 13

Top deleted words:

- `a`: 166
- `and`: 49
- `to`: 38
- `in`: 37
- `the`: 32
- `i`: 29
- `you`: 17
- `he`: 16
- `we`: 15
- `were`: 14
- `there`: 13
- `all`: 13
- `it`: 12
- `on`: 12
- `an`: 12
- `had`: 11
- `are`: 11
- `of`: 11
- `be`: 11
- `for`: 10
- `is`: 10
- `some`: 10
- `see`: 10
- `with`: 10
- `can`: 9
- `how`: 9
- `what`: 9
- `his`: 8
- `was`: 8
- `been`: 7

Top inserted words:

- `a`: 71
- `in`: 35
- `an`: 32
- `and`: 22
- `the`: 22
- `be`: 19
- `to`: 18
- `i`: 15
- `over`: 12
- `for`: 11
- `o`: 11
- `al`: 10
- `of`: 9
- `t`: 9
- `is`: 8
- `with`: 8
- `e`: 8
- `it`: 8
- `ar`: 8
- `at`: 8
- `any`: 8
- `he`: 8
- `some`: 8
- `as`: 7
- `there`: 7
- `s`: 6
- `on`: 6
- `that`: 5
- `man`: 5
- `his`: 5

Duration-bucket WER:

- 5-10s: 0.4311
- <5s: 0.5563
- 10-15s: 0.4277

Highest speaker-level WER:

- 1272: 0.6010
- 6313: 0.5720
- 777: 0.5689
- 5338: 0.5303
- 5536: 0.5261
- 2803: 0.5248
- 1988: 0.5080
- 5694: 0.5020
- 3853: 0.5004
- 2412: 0.4868
- 6241: 0.4826
- 174: 0.4775
- 3752: 0.4739
- 422: 0.4706
- 8842: 0.4697
- 251: 0.4662
- 1673: 0.4635
- 652: 0.4606
- 1919: 0.4600
- 2902: 0.4590
- 7850: 0.4550
- 3081: 0.4435
- 2078: 0.4416
- 7976: 0.4379
- 6295: 0.4359
- 1462: 0.4353
- 5895: 0.4322
- 2086: 0.4280
- 3000: 0.4275
- 2277: 0.4205

Representative high-error utterances:

- `2078-142845-0009` ref: illustration italian millet / hyp: i lu stration ad ctay an melt
- `3081-166546-0021` ref: i emphasised complacently / hyp: a am thesiset com pliy son sli
- `5694-64038-0000` ref: advance into tennessee / hyp: at vance int a ten a s
- `1919-142785-0039` ref: illustration marjoram / hyp: i istration maridur im
- `2078-142845-0044` ref: italian rusks / hyp: e tay in resx

## e13_layer4_bilstm_ctc

Top substitutions:

- `and` → `in`: 85
- `and` → `an`: 70
- `in` → `and`: 60
- `in` → `an`: 48
- `an` → `and`: 32
- `it` → `at`: 32
- `is` → `as`: 29
- `of` → `a`: 28
- `are` → `ar`: 28
- `his` → `is`: 28
- `too` → `to`: 26
- `there` → `ther`: 24
- `of` → `o`: 23
- `thee` → `the`: 23
- `is` → `his`: 22
- `the` → `tha`: 20
- `at` → `it`: 19
- `it` → `i`: 19
- `into` → `to`: 18
- `their` → `ther`: 18
- `as` → `is`: 18
- `been` → `ben`: 17
- `know` → `no`: 17
- `the` → `he`: 17
- `the` → `te`: 16
- `in` → `ind`: 16
- `our` → `ar`: 16
- `two` → `to`: 14
- `all` → `al`: 14
- `that` → `the`: 14

Top deleted words:

- `a`: 176
- `in`: 41
- `the`: 39
- `and`: 36
- `to`: 35
- `i`: 32
- `you`: 23
- `is`: 18
- `for`: 17
- `he`: 14
- `of`: 13
- `be`: 13
- `it`: 12
- `not`: 12
- `no`: 12
- `at`: 12
- `her`: 11
- `why`: 11
- `an`: 11
- `shall`: 10
- `are`: 10
- `all`: 10
- `some`: 10
- `but`: 10
- `do`: 10
- `we`: 10
- `his`: 9
- `with`: 9
- `am`: 9
- `has`: 8

Top inserted words:

- `a`: 110
- `in`: 49
- `o`: 26
- `and`: 25
- `an`: 24
- `to`: 23
- `the`: 22
- `i`: 20
- `some`: 14
- `for`: 14
- `be`: 13
- `on`: 13
- `of`: 13
- `e`: 13
- `is`: 12
- `s`: 11
- `as`: 11
- `with`: 10
- `at`: 9
- `all`: 9
- `it`: 9
- `t`: 7
- `mount`: 7
- `any`: 7
- `ar`: 7
- `you`: 7
- `were`: 6
- `ther`: 6
- `what`: 6
- `ad`: 6

Duration-bucket WER:

- 10-15s: 0.4322
- <5s: 0.5631
- 5-10s: 0.4599
- >=15s: 0.4232

Highest speaker-level WER:

- 3570: 0.5712
- 1995: 0.5626
- 8555: 0.5446
- 8463: 0.5347
- 4507: 0.5281
- 7176: 0.5229
- 4992: 0.5171
- 3575: 0.5093
- 2961: 0.5078
- 4446: 0.5046
- 2300: 0.5000
- 7729: 0.4970
- 260: 0.4945
- 8455: 0.4942
- 4077: 0.4938
- 6829: 0.4922
- 5142: 0.4922
- 2830: 0.4755
- 237: 0.4748
- 2094: 0.4653
- 61: 0.4646
- 121: 0.4617
- 5683: 0.4581
- 6930: 0.4574
- 4970: 0.4569
- 1188: 0.4560
- 7127: 0.4395
- 5105: 0.4375
- 3729: 0.4240
- 5639: 0.4236

Representative high-error utterances:

- `3575-170457-0016` ref: farewell madam / hyp: fir o e onm
- `8555-292519-0002` ref: venice / hyp: tha is
- `1089-134691-0020` ref: hello stephanos here comes the dedalus / hyp: had ho o stipen os her com ws the deat lis
- `5105-28241-0014` ref: another circumstance was most remarkable / hyp: ando the saick and sants was mos fr macuble
- `1089-134691-0018` ref: again again / hyp: agan a gan

## e13_layer5_bilstm_ctc

Top substitutions:

- `and` → `an`: 71
- `and` → `in`: 66
- `in` → `an`: 58
- `in` → `and`: 54
- `is` → `as`: 31
- `it` → `at`: 30
- `into` → `to`: 23
- `are` → `ar`: 21
- `his` → `is`: 20
- `is` → `his`: 17
- `a` → `o`: 16
- `too` → `two`: 14
- `at` → `it`: 14
- `a` → `the`: 14
- `been` → `ben`: 13
- `and` → `ind`: 13
- `our` → `ar`: 13
- `all` → `al`: 13
- `an` → `and`: 13
- `they` → `the`: 12
- `the` → `a`: 12
- `of` → `a`: 12
- `where` → `were`: 12
- `can` → `con`: 11
- `two` → `to`: 11
- `his` → `as`: 11
- `of` → `o`: 11
- `it's` → `its`: 11
- `a` → `at`: 11
- `there` → `their`: 11

Top deleted words:

- `a`: 114
- `in`: 39
- `and`: 32
- `to`: 29
- `i`: 26
- `the`: 21
- `be`: 17
- `an`: 13
- `of`: 12
- `it`: 12
- `are`: 11
- `is`: 11
- `were`: 10
- `all`: 10
- `he`: 10
- `was`: 9
- `had`: 8
- `see`: 8
- `do`: 7
- `you`: 7
- `some`: 7
- `or`: 7
- `if`: 7
- `we`: 7
- `but`: 7
- `can`: 7
- `time`: 6
- `what`: 6
- `she`: 5
- `this`: 5

Top inserted words:

- `a`: 74
- `in`: 42
- `to`: 25
- `an`: 22
- `i`: 20
- `and`: 17
- `the`: 16
- `with`: 11
- `be`: 11
- `for`: 11
- `o`: 10
- `t`: 9
- `some`: 9
- `he`: 9
- `over`: 9
- `al`: 8
- `e`: 8
- `s`: 8
- `of`: 8
- `any`: 8
- `all`: 6
- `at`: 6
- `no`: 6
- `on`: 6
- `it`: 6
- `un`: 6
- `ad`: 5
- `ac`: 5
- `de`: 5
- `you`: 5

Duration-bucket WER:

- 5-10s: 0.3544
- <5s: 0.5013
- 10-15s: 0.3437

Highest speaker-level WER:

- 1272: 0.5055
- 777: 0.4878
- 6313: 0.4599
- 2803: 0.4515
- 5536: 0.4500
- 1988: 0.4477
- 5694: 0.4466
- 5338: 0.4227
- 2412: 0.4166
- 3853: 0.4165
- 6241: 0.4109
- 1919: 0.4107
- 8842: 0.4014
- 652: 0.4006
- 2902: 0.3979
- 1673: 0.3975
- 174: 0.3912
- 251: 0.3899
- 3081: 0.3857
- 7850: 0.3831
- 2078: 0.3792
- 422: 0.3765
- 6319: 0.3702
- 5895: 0.3676
- 8297: 0.3674
- 3000: 0.3644
- 7976: 0.3643
- 3752: 0.3626
- 2086: 0.3613
- 1462: 0.3596

Representative high-error utterances:

- `1462-170138-0003` ref: alexander exclaimed mildly / hyp: ill as ender exclimed to moudly
- `1919-142785-0039` ref: illustration marjoram / hyp: i lstration mar tuerom
- `2078-142845-0009` ref: illustration italian millet / hyp: i l sterasion attaig and mealet
- `777-126732-0081` ref: comfortable dear / hyp: of to le ter
- `1462-170145-0021` ref: alexander flushed angrily / hyp: il es ender foushed angrlly

## e13_layer5_bilstm_ctc

Top substitutions:

- `and` → `an`: 93
- `in` → `and`: 53
- `and` → `in`: 52
- `in` → `an`: 42
- `is` → `as`: 31
- `it` → `at`: 29
- `an` → `and`: 28
- `is` → `his`: 25
- `too` → `two`: 22
- `thee` → `the`: 21
- `his` → `is`: 19
- `are` → `ar`: 18
- `and` → `ind`: 18
- `of` → `o`: 18
- `of` → `a`: 16
- `at` → `it`: 16
- `know` → `no`: 16
- `there` → `ther`: 15
- `in` → `ind`: 15
- `this` → `the`: 15
- `their` → `ther`: 14
- `been` → `ben`: 13
- `the` → `a`: 13
- `too` → `to`: 13
- `into` → `to`: 12
- `will` → `wil`: 12
- `than` → `then`: 12
- `as` → `is`: 12
- `know` → `now`: 12
- `two` → `too`: 11

Top deleted words:

- `a`: 138
- `the`: 40
- `in`: 39
- `and`: 29
- `to`: 24
- `i`: 24
- `you`: 17
- `for`: 16
- `on`: 16
- `it`: 15
- `be`: 14
- `he`: 14
- `is`: 13
- `her`: 13
- `or`: 13
- `of`: 13
- `at`: 12
- `can`: 11
- `do`: 10
- `there`: 9
- `had`: 9
- `as`: 9
- `an`: 9
- `his`: 8
- `all`: 8
- `we`: 8
- `will`: 7
- `why`: 7
- `him`: 6
- `more`: 6

Top inserted words:

- `a`: 104
- `in`: 49
- `to`: 23
- `be`: 21
- `an`: 21
- `and`: 20
- `i`: 18
- `o`: 16
- `the`: 14
- `some`: 12
- `as`: 11
- `ad`: 11
- `at`: 10
- `on`: 10
- `all`: 9
- `any`: 9
- `e`: 9
- `he`: 8
- `you`: 7
- `every`: 7
- `over`: 7
- `with`: 7
- `al`: 7
- `te`: 6
- `out`: 6
- `never`: 6
- `for`: 6
- `of`: 6
- `re`: 6
- `where`: 5

Duration-bucket WER:

- 10-15s: 0.3487
- <5s: 0.5137
- 5-10s: 0.3891
- >=15s: 0.3336

Highest speaker-level WER:

- 1995: 0.4851
- 3570: 0.4774
- 8555: 0.4703
- 7176: 0.4628
- 8463: 0.4476
- 4507: 0.4469
- 7729: 0.4466
- 2961: 0.4446
- 260: 0.4413
- 2300: 0.4394
- 6829: 0.4321
- 4992: 0.4294
- 4077: 0.4290
- 3575: 0.4248
- 8455: 0.4142
- 2830: 0.4062
- 4446: 0.3993
- 5142: 0.3964
- 61: 0.3903
- 5683: 0.3849
- 2094: 0.3811
- 1188: 0.3804
- 237: 0.3763
- 6930: 0.3754
- 4970: 0.3745
- 5105: 0.3630
- 7127: 0.3621
- 121: 0.3594
- 1284: 0.3582
- 3729: 0.3491

Representative high-error utterances:

- `8555-292519-0002` ref: venice / hyp: twe e e
- `1089-134691-0003` ref: the university / hyp: te un of vrsfitay
- `1089-134691-0024` ref: stephanos dedalos / hyp: stuff in osdead oaus
- `3575-170457-0016` ref: farewell madam / hyp: fi bo e dom
- `5105-28241-0014` ref: another circumstance was most remarkable / hyp: and he thi shack on sants wuas mos thr arcoable

## e13_layer7_bilstm_ctc

Top substitutions:

- `and` → `an`: 55
- `in` → `and`: 44
- `and` → `in`: 34
- `in` → `an`: 23
- `in` → `ind`: 13
- `a` → `the`: 12
- `it` → `at`: 10
- `bartley` → `bartly`: 9
- `is` → `as`: 9
- `this` → `the`: 9
- `the` → `a`: 8
- `his` → `is`: 8
- `of` → `o`: 8
- `add` → `ad`: 8
- `been` → `ben`: 8
- `four` → `for`: 8
- `into` → `to`: 8
- `and` → `ad`: 8
- `randal` → `randle`: 8
- `too` → `to`: 7
- `mary` → `marry`: 7
- `where` → `were`: 7
- `asleep` → `sleep`: 6
- `then` → `than`: 6
- `and` → `ind`: 6
- `o` → `oh`: 6
- `eggs` → `egs`: 6
- `is` → `his`: 6
- `are` → `ar`: 6
- `to` → `a`: 6

Top deleted words:

- `a`: 51
- `the`: 19
- `in`: 18
- `and`: 14
- `to`: 14
- `i`: 9
- `is`: 8
- `do`: 6
- `will`: 6
- `of`: 5
- `it`: 5
- `he`: 5
- `some`: 5
- `an`: 5
- `any`: 4
- `our`: 4
- `we`: 4
- `or`: 4
- `on`: 4
- `this`: 3
- `they`: 3
- `with`: 3
- `you`: 3
- `mary`: 3
- `well`: 3
- `up`: 3
- `but`: 3
- `there`: 3
- `every`: 3
- `no`: 3

Top inserted words:

- `a`: 50
- `in`: 15
- `the`: 14
- `i`: 13
- `and`: 9
- `to`: 8
- `for`: 7
- `of`: 6
- `o`: 6
- `he`: 6
- `an`: 5
- `it`: 5
- `mag`: 5
- `grand`: 4
- `some`: 4
- `with`: 4
- `foot`: 4
- `mean`: 4
- `there`: 4
- `any`: 4
- `s`: 3
- `is`: 3
- `sweet`: 3
- `im`: 3
- `where`: 3
- `we`: 3
- `all`: 3
- `al`: 3
- `en`: 3
- `t`: 3

Duration-bucket WER:

- 5-10s: 0.1893
- <5s: 0.3400
- 10-15s: 0.1793

Highest speaker-level WER:

- 777: 0.3278
- 1272: 0.2886
- 6313: 0.2872
- 652: 0.2752
- 5694: 0.2609
- 1919: 0.2567
- 1988: 0.2524
- 6241: 0.2484
- 5536: 0.2465
- 2078: 0.2432
- 1673: 0.2422
- 3853: 0.2355
- 8842: 0.2344
- 3081: 0.2330
- 2803: 0.2325
- 7850: 0.2275
- 5338: 0.2270
- 174: 0.2250
- 1462: 0.2239
- 5895: 0.2214
- 251: 0.2212
- 2412: 0.2136
- 2428: 0.2134
- 2086: 0.2126
- 3752: 0.2125
- 6319: 0.2092
- 2902: 0.2059
- 84: 0.2053
- 6295: 0.1940
- 3000: 0.1929

Representative high-error utterances:

- `2078-142845-0009` ref: illustration italian millet / hyp: il stasion i tai and melet
- `3081-166546-0021` ref: i emphasised complacently / hyp: a am the s ist complasently
- `777-126732-0081` ref: comfortable dear / hyp: coft i tha ther
- `8842-304647-0007` ref: most wonderful / hyp: mos  mn or fo
- `5536-43358-0016` ref: who may condemn his superstition / hyp: whe nay cand da in hes sou brs taston

## e13_layer7_bilstm_ctc

Top substitutions:

- `and` → `in`: 50
- `and` → `an`: 50
- `in` → `and`: 37
- `in` → `an`: 32
- `is` → `as`: 20
- `an` → `and`: 17
- `thee` → `the`: 14
- `the` → `a`: 12
- `of` → `a`: 12
- `are` → `ar`: 11
- `it` → `at`: 11
- `bartley` → `bartly`: 11
- `too` → `to`: 10
- `this` → `the`: 10
- `this` → `thes`: 10
- `a` → `the`: 10
- `christ` → `crist`: 10
- `consumption` → `consemption`: 10
- `and` → `ind`: 9
- `is` → `his`: 9
- `in` → `ind`: 9
- `true` → `tru`: 9
- `a` → `o`: 9
- `they` → `the`: 8
- `his` → `is`: 8
- `to` → `o`: 8
- `you` → `ou`: 8
- `know` → `kno`: 8
- `two` → `to`: 7
- `all` → `al`: 7

Top deleted words:

- `a`: 45
- `in`: 19
- `the`: 18
- `i`: 17
- `to`: 16
- `and`: 13
- `or`: 11
- `of`: 10
- `all`: 8
- `do`: 8
- `he`: 7
- `is`: 7
- `can`: 6
- `it`: 6
- `more`: 6
- `be`: 6
- `but`: 6
- `for`: 5
- `some`: 5
- `know`: 4
- `so`: 4
- `at`: 4
- `paul`: 4
- `when`: 4
- `what`: 4
- `new`: 4
- `well`: 3
- `you`: 3
- `have`: 3
- `her`: 3

Top inserted words:

- `a`: 42
- `in`: 26
- `i`: 17
- `the`: 15
- `o`: 10
- `and`: 10
- `be`: 9
- `s`: 9
- `an`: 9
- `some`: 8
- `it`: 7
- `every`: 6
- `to`: 6
- `is`: 6
- `all`: 6
- `never`: 6
- `of`: 5
- `house`: 5
- `any`: 5
- `with`: 5
- `there`: 5
- `man`: 5
- `up`: 4
- `as`: 4
- `for`: 4
- `on`: 4
- `after`: 3
- `under`: 3
- `govenor`: 3
- `com`: 3

Duration-bucket WER:

- 10-15s: 0.1915
- <5s: 0.3596
- 5-10s: 0.2161
- >=15s: 0.1828

Highest speaker-level WER:

- 1995: 0.3177
- 7176: 0.3113
- 3570: 0.3099
- 8555: 0.2949
- 2961: 0.2829
- 4077: 0.2778
- 7729: 0.2772
- 4507: 0.2729
- 8463: 0.2710
- 4992: 0.2630
- 3575: 0.2564
- 260: 0.2551
- 4446: 0.2490
- 2300: 0.2422
- 2830: 0.2352
- 4970: 0.2315
- 6829: 0.2309
- 2094: 0.2304
- 5142: 0.2293
- 61: 0.2282
- 8455: 0.2246
- 1188: 0.2245
- 908: 0.2214
- 6930: 0.2128
- 121: 0.2117
- 1284: 0.2081
- 5105: 0.2064
- 7127: 0.2055
- 237: 0.2050
- 1089: 0.2005

Representative high-error utterances:

- `3575-170457-0016` ref: farewell madam / hyp: fee ao bet am
- `8555-292519-0002` ref: venice / hyp: i a
- `8463-294825-0000` ref: it's almost beyond conjecture / hyp: it so mos to be ond conjectire
- `1089-134691-0020` ref: hello stephanos here comes the dedalus / hyp: hed o os s of an nos here coumems the diti us
- `1995-1836-0011` ref: positively heroic added cresswell avoiding his sister's eyes / hyp: bos i eaemt nhar alit a it rs werl worting es isters as

## e13_layer8_bilstm_ctc

Top substitutions:

- `and` → `an`: 40
- `in` → `and`: 22
- `and` → `in`: 19
- `in` → `an`: 15
- `in` → `ind`: 12
- `a` → `the`: 10
- `the` → `a`: 9
- `bread` → `bred`: 8
- `it's` → `its`: 7
- `this` → `the`: 7
- `of` → `o`: 7
- `randal` → `randle`: 7
- `is` → `as`: 6
- `add` → `ad`: 6
- `to` → `o`: 6
- `now` → `no`: 5
- `oh` → `o`: 5
- `mary` → `marry`: 5
- `the` → `th`: 5
- `o` → `oh`: 5
- `eggs` → `egs`: 5
- `you're` → `your`: 5
- `and` → `ad`: 5
- `on` → `an`: 5
- `phoebe` → `feby`: 5
- `carrie` → `carry`: 5
- `two` → `to`: 5
- `a` → `at`: 5
- `milner` → `millner`: 5
- `bozzle` → `basl`: 5

Top deleted words:

- `a`: 35
- `in`: 19
- `to`: 14
- `i`: 12
- `and`: 10
- `the`: 7
- `this`: 5
- `do`: 5
- `you`: 5
- `don`: 5
- `it`: 4
- `very`: 4
- `any`: 4
- `of`: 3
- `white`: 3
- `will`: 3
- `every`: 3
- `some`: 3
- `on`: 3
- `shall`: 3
- `is`: 3
- `that`: 3
- `an`: 3
- `where`: 2
- `mac`: 2
- `little`: 2
- `such`: 2
- `serve`: 2
- `at`: 2
- `as`: 2

Top inserted words:

- `a`: 37
- `in`: 15
- `the`: 12
- `i`: 9
- `any`: 7
- `of`: 5
- `is`: 5
- `grand`: 5
- `he`: 5
- `and`: 4
- `do`: 4
- `ad`: 4
- `all`: 4
- `be`: 4
- `an`: 4
- `up`: 3
- `s`: 3
- `some`: 3
- `te`: 3
- `with`: 3
- `al`: 3
- `to`: 3
- `no`: 3
- `ever`: 3
- `mag`: 3
- `bus`: 2
- `fire`: 2
- `o`: 2
- `mo`: 2
- `lo`: 2

Duration-bucket WER:

- 5-10s: 0.1539
- <5s: 0.2873
- 10-15s: 0.1450

Highest speaker-level WER:

- 777: 0.2697
- 652: 0.2429
- 1272: 0.2388
- 6313: 0.2234
- 1673: 0.2132
- 5694: 0.2119
- 1919: 0.2105
- 251: 0.2088
- 1988: 0.2082
- 6241: 0.2072
- 5895: 0.2028
- 8842: 0.1938
- 3752: 0.1935
- 5536: 0.1927
- 3853: 0.1905
- 7850: 0.1835
- 2803: 0.1828
- 5338: 0.1781
- 3081: 0.1777
- 2086: 0.1773
- 84: 0.1771
- 2035: 0.1758
- 2412: 0.1728
- 2902: 0.1728
- 422: 0.1725
- 2428: 0.1713
- 2078: 0.1712
- 174: 0.1650
- 6319: 0.1632
- 1462: 0.1622

Representative high-error utterances:

- `2078-142845-0009` ref: illustration italian millet / hyp: ill stration at ti and melit
- `777-126732-0081` ref: comfortable dear / hyp: copth o la dere
- `2078-142845-0048` ref: seventeen thirty four / hyp: si hin teen thirty f war
- `2428-83699-0041` ref: i'm mister christopher from london / hyp: i mes a cres af fer fom lntin
- `1919-142785-0048` ref: french forcemeat / hyp: frinte foris meat

## e13_layer8_bilstm_ctc

Top substitutions:

- `and` → `an`: 37
- `and` → `in`: 31
- `in` → `and`: 29
- `in` → `an`: 24
- `paul` → `pall`: 13
- `an` → `and`: 11
- `a` → `the`: 10
- `christ` → `crist`: 10
- `oh` → `o`: 9
- `too` → `to`: 9
- `the` → `he`: 9
- `the` → `a`: 9
- `consumption` → `consemption`: 9
- `thee` → `the`: 9
- `is` → `as`: 8
- `this` → `the`: 8
- `i` → `y`: 8
- `two` → `to`: 7
- `and` → `nd`: 7
- `o` → `oh`: 7
- `the` → `ha`: 7
- `is` → `his`: 7
- `ruth` → `roth`: 7
- `robin` → `robbin`: 7
- `kenneth` → `kenith`: 7
- `their` → `ther`: 6
- `wholly` → `holy`: 6
- `are` → `ar`: 6
- `a` → `an`: 6
- `and` → `ind`: 6

Top deleted words:

- `a`: 42
- `in`: 12
- `i`: 10
- `and`: 10
- `he`: 9
- `to`: 9
- `the`: 8
- `this`: 7
- `more`: 6
- `is`: 6
- `of`: 6
- `it`: 6
- `at`: 5
- `her`: 5
- `all`: 5
- `some`: 5
- `why`: 5
- `on`: 5
- `have`: 4
- `not`: 4
- `be`: 4
- `we`: 4
- `or`: 4
- `an`: 4
- `do`: 4
- `school`: 3
- `no`: 3
- `any`: 3
- `time`: 3
- `but`: 3

Top inserted words:

- `a`: 35
- `the`: 15
- `in`: 13
- `i`: 13
- `s`: 11
- `some`: 10
- `an`: 9
- `to`: 8
- `he`: 7
- `be`: 6
- `every`: 6
- `it`: 6
- `up`: 5
- `t`: 5
- `o`: 5
- `and`: 5
- `any`: 5
- `r`: 5
- `as`: 4
- `at`: 4
- `fire`: 4
- `all`: 4
- `with`: 4
- `al`: 4
- `for`: 4
- `th`: 4
- `can`: 4
- `never`: 4
- `un`: 3
- `after`: 3

Duration-bucket WER:

- 10-15s: 0.1573
- <5s: 0.3091
- 5-10s: 0.1862
- >=15s: 0.1517

Highest speaker-level WER:

- 1995: 0.2973
- 8555: 0.2704
- 7176: 0.2594
- 3570: 0.2586
- 4507: 0.2562
- 2961: 0.2500
- 8463: 0.2411
- 7729: 0.2407
- 4992: 0.2281
- 2300: 0.2242
- 4446: 0.2098
- 4077: 0.2083
- 5142: 0.2042
- 3575: 0.2042
- 908: 0.2022
- 260: 0.2003
- 6829: 0.1985
- 1188: 0.1960
- 8455: 0.1919
- 4970: 0.1888
- 2094: 0.1879
- 61: 0.1843
- 5105: 0.1842
- 1284: 0.1811
- 2830: 0.1782
- 1089: 0.1748
- 121: 0.1744
- 237: 0.1741
- 7127: 0.1739
- 6930: 0.1633

Representative high-error utterances:

- `1089-134691-0003` ref: the university / hyp: they youn av varity
- `1089-134691-0024` ref: stephanos dedalos / hyp: s afrnows deat los
- `3575-170457-0016` ref: farewell madam / hyp: fee ao me em
- `3729-6852-0010` ref: i never had any family / hyp: gy nhe hrm h e henee rove he tey
- `1995-1836-0011` ref: positively heroic added cresswell avoiding his sister's eyes / hyp: os i a ne har all adit couros wa wortiyg he cis tirs is

## e14_layer_mixture_bilstm_ctc

Top substitutions:

- `and` → `an`: 34
- `in` → `and`: 31
- `and` → `in`: 20
- `in` → `an`: 16
- `a` → `the`: 12
- `too` → `to`: 12
- `in` → `ind`: 10
- `it's` → `its`: 9
- `into` → `to`: 9
- `add` → `ad`: 9
- `four` → `for`: 9
- `are` → `ar`: 8
- `bread` → `bred`: 8
- `phoebe` → `feeby`: 8
- `an` → `and`: 8
- `the` → `a`: 7
- `asleep` → `sleep`: 7
- `mary` → `marry`: 7
- `o` → `oh`: 7
- `eggs` → `egs`: 7
- `to` → `o`: 6
- `one` → `wone`: 6
- `who's` → `whose`: 6
- `good` → `god`: 6
- `that` → `tha`: 6
- `does` → `dos`: 6
- `on` → `an`: 6
- `either` → `ither`: 6
- `of` → `o`: 6
- `two` → `to`: 5

Top deleted words:

- `a`: 43
- `in`: 16
- `and`: 12
- `the`: 10
- `to`: 10
- `i`: 9
- `had`: 8
- `an`: 6
- `mary`: 5
- `north`: 5
- `this`: 5
- `how`: 4
- `of`: 4
- `do`: 4
- `any`: 4
- `one`: 4
- `dead`: 4
- `under`: 3
- `it`: 3
- `all`: 3
- `will`: 3
- `would`: 3
- `some`: 3
- `out`: 3
- `on`: 3
- `was`: 3
- `are`: 3
- `where`: 2
- `mac`: 2
- `through`: 2

Top inserted words:

- `a`: 44
- `in`: 15
- `the`: 13
- `any`: 8
- `an`: 7
- `at`: 6
- `o`: 5
- `grand`: 5
- `some`: 5
- `and`: 4
- `to`: 4
- `i`: 4
- `for`: 4
- `it`: 4
- `he`: 4
- `al`: 4
- `dis`: 3
- `with`: 3
- `be`: 3
- `of`: 3
- `t`: 3
- `table`: 3
- `all`: 3
- `maine`: 2
- `hil`: 2
- `s`: 2
- `on`: 2
- `under`: 2
- `out`: 2
- `garrow`: 2

Duration-bucket WER:

- 5-10s: 0.1696
- <5s: 0.2944
- 10-15s: 0.1588

Highest speaker-level WER:

- 777: 0.2798
- 652: 0.2524
- 1272: 0.2517
- 5694: 0.2466
- 1988: 0.2355
- 6313: 0.2340
- 1919: 0.2320
- 5895: 0.2259
- 1673: 0.2236
- 2078: 0.2224
- 6241: 0.2221
- 251: 0.2221
- 5536: 0.2076
- 2803: 0.2065
- 8842: 0.2050
- 3752: 0.2014
- 2902: 0.1990
- 2086: 0.1964
- 174: 0.1963
- 3081: 0.1907
- 3853: 0.1905
- 2428: 0.1893
- 84: 0.1889
- 7850: 0.1835
- 6295: 0.1829
- 2412: 0.1804
- 6319: 0.1796
- 5338: 0.1791
- 2277: 0.1726
- 2035: 0.1722

Representative high-error utterances:

- `8842-304647-0007` ref: most wonderful / hyp: mo sf mne or fal
- `1919-142785-0048` ref: french forcemeat / hyp: frins for is meet
- `2428-83705-0036` ref: someone sniggered / hyp: som on s negured
- `3081-166546-0021` ref: i emphasised complacently / hyp: a am this ist compla sently
- `2078-142845-0009` ref: illustration italian millet / hyp: illstration at tel end melat

## e14_layer_mixture_bilstm_ctc

Top substitutions:

- `and` → `an`: 34
- `and` → `in`: 31
- `in` → `and`: 25
- `in` → `an`: 25
- `an` → `and`: 17
- `too` → `to`: 15
- `paul` → `pall`: 13
- `is` → `as`: 11
- `thee` → `the`: 11
- `is` → `his`: 10
- `in` → `ind`: 10
- `will` → `wil`: 9
- `a` → `the`: 9
- `consumption` → `consemption`: 9
- `two` → `to`: 8
- `the` → `a`: 8
- `this` → `the`: 8
- `oh` → `o`: 8
- `wholly` → `holy`: 7
- `and` → `ind`: 7
- `o` → `oh`: 7
- `shone` → `shown`: 7
- `all` → `al`: 7
- `this` → `thes`: 7
- `where` → `were`: 7
- `memory` → `memery`: 7
- `sir` → `ser`: 7
- `red` → `read`: 6
- `i'm` → `im`: 6
- `there` → `ther`: 6

Top deleted words:

- `a`: 44
- `i`: 16
- `in`: 15
- `to`: 13
- `and`: 11
- `is`: 9
- `the`: 9
- `or`: 8
- `he`: 7
- `more`: 6
- `all`: 6
- `this`: 5
- `was`: 5
- `be`: 5
- `do`: 5
- `it`: 4
- `school`: 4
- `but`: 4
- `for`: 4
- `at`: 4
- `why`: 4
- `new`: 4
- `well`: 3
- `then`: 3
- `that`: 3
- `know`: 3
- `any`: 3
- `up`: 3
- `mister`: 3
- `of`: 3

Top inserted words:

- `a`: 33
- `in`: 14
- `the`: 14
- `i`: 10
- `o`: 9
- `s`: 7
- `de`: 6
- `every`: 6
- `any`: 6
- `an`: 6
- `house`: 5
- `some`: 5
- `fire`: 5
- `is`: 5
- `never`: 5
- `be`: 4
- `t`: 4
- `what`: 4
- `all`: 4
- `for`: 4
- `other`: 4
- `he`: 4
- `where`: 3
- `there`: 3
- `to`: 3
- `on`: 3
- `up`: 3
- `it`: 3
- `here`: 3
- `and`: 3

Duration-bucket WER:

- 10-15s: 0.1648
- <5s: 0.3219
- 5-10s: 0.1899
- >=15s: 0.1640

Highest speaker-level WER:

- 1995: 0.2919
- 7176: 0.2819
- 8555: 0.2660
- 2961: 0.2647
- 3570: 0.2640
- 8463: 0.2476
- 7729: 0.2372
- 4507: 0.2365
- 4992: 0.2311
- 4077: 0.2253
- 3575: 0.2242
- 4446: 0.2235
- 2300: 0.2226
- 5142: 0.2132
- 260: 0.2128
- 2094: 0.2103
- 8455: 0.2086
- 4970: 0.2082
- 2830: 0.2063
- 908: 0.2059
- 6829: 0.2053
- 1188: 0.1952
- 121: 0.1895
- 5105: 0.1888
- 237: 0.1878
- 7127: 0.1850
- 61: 0.1816
- 1089: 0.1812
- 8224: 0.1808
- 1284: 0.1771

Representative high-error utterances:

- `1089-134691-0003` ref: the university / hyp: thiy yeun ho vrceity
- `2094-142345-0016` ref: spinning indeed / hyp: sp le ain vheed
- `1995-1836-0011` ref: positively heroic added cresswell avoiding his sister's eyes / hyp: o thi a la h alit t curest wl a wartig hes sistirs aois
- `5105-28241-0010` ref: ocean reigned supreme / hyp: ashen raimnt s stor prame
- `5105-28241-0014` ref: another circumstance was most remarkable / hyp: an ot heseirccan sa ent swous mous rmackable

## e15a_linear_layer8_ctc

Top substitutions:

- `and` → `an`: 178
- `all` → `al`: 97
- `the` → `he`: 83
- `there` → `ther`: 70
- `been` → `ben`: 68
- `were` → `wer`: 66
- `one` → `on`: 60
- `are` → `ar`: 57
- `their` → `ther`: 55
- `his` → `is`: 53
- `with` → `wit`: 50
- `some` → `som`: 48
- `one` → `won`: 46
- `made` → `mad`: 45
- `it` → `i`: 44
- `the` → `te`: 40
- `more` → `mor`: 37
- `you` → `yo`: 36
- `were` → `we`: 36
- `have` → `hav`: 36
- `who` → `ho`: 34
- `now` → `n`: 34
- `all` → `l`: 33
- `they` → `the`: 33
- `like` → `lik`: 33
- `the` → `th`: 32
- `these` → `thes`: 32
- `little` → `litl`: 32
- `said` → `sad`: 31
- `would` → `wuld`: 31

Top deleted words:

- `the`: 148
- `a`: 95
- `and`: 66
- `i`: 48
- `it`: 40
- `in`: 39
- `to`: 38
- `he`: 36
- `with`: 34
- `you`: 32
- `his`: 31
- `of`: 31
- `but`: 30
- `not`: 26
- `that`: 26
- `they`: 25
- `had`: 25
- `their`: 25
- `all`: 25
- `at`: 24
- `was`: 23
- `there`: 21
- `one`: 19
- `do`: 19
- `is`: 19
- `this`: 18
- `said`: 17
- `have`: 16
- `would`: 16
- `time`: 16

Top inserted words:

- `a`: 36
- `un`: 17
- `in`: 15
- `al`: 8
- `be`: 8
- `pr`: 8
- `l`: 7
- `an`: 7
- `ben`: 6
- `som`: 6
- `to`: 6
- `re`: 5
- `the`: 5
- `he`: 5
- `any`: 5
- `my`: 5
- `is`: 5
- `te`: 4
- `o`: 4
- `on`: 4
- `i`: 4
- `ther`: 3
- `grnd`: 3
- `u`: 3
- `over`: 3
- `ros`: 3
- `every`: 3
- `what`: 3
- `hr`: 3
- `some`: 3

Duration-bucket WER:

- 5-10s: 0.6287
- <5s: 0.7047
- 10-15s: 0.6231

Highest speaker-level WER:

- 1272: 0.7244
- 5694: 0.7020
- 777: 0.6887
- 6241: 0.6870
- 1988: 0.6793
- 6313: 0.6752
- 652: 0.6751
- 2086: 0.6740
- 5338: 0.6703
- 2412: 0.6649
- 2803: 0.6569
- 1919: 0.6550
- 3000: 0.6520
- 8842: 0.6514
- 1462: 0.6513
- 8297: 0.6493
- 251: 0.6473
- 3752: 0.6453
- 7976: 0.6431
- 84: 0.6418
- 7850: 0.6403
- 3853: 0.6398
- 5536: 0.6352
- 1673: 0.6327
- 2078: 0.6320
- 5895: 0.6298
- 422: 0.6294
- 6345: 0.6243
- 3576: 0.6240
- 3081: 0.6230

Representative high-error utterances:

- `2078-142845-0044` ref: italian rusks / hyp: a tin rs
- `3081-166546-0005` ref: george nodded / hyp: tjrch nod adddn
- `777-126732-0017` ref: very characteristic perfectly typical / hyp: fcar torstip e le te piccl
- `1462-170145-0021` ref: alexander flushed angrily / hyp: l sndr fld angrly
- `251-137823-0024` ref: tom nodded unhappily / hyp: tm nted un haply

## e15a_linear_layer8_ctc

Top substitutions:

- `and` → `an`: 206
- `all` → `al`: 107
- `the` → `he`: 81
- `were` → `wer`: 81
- `are` → `ar`: 80
- `been` → `ben`: 76
- `with` → `wit`: 74
- `there` → `ther`: 73
- `their` → `ther`: 68
- `more` → `mor`: 64
- `one` → `won`: 55
- `the` → `te`: 50
- `have` → `hav`: 48
- `like` → `lik`: 48
- `who` → `ho`: 47
- `one` → `on`: 42
- `some` → `som`: 41
- `great` → `grat`: 41
- `all` → `l`: 41
- `will` → `wl`: 40
- `little` → `litl`: 39
- `you` → `yo`: 39
- `the` → `th`: 37
- `said` → `sad`: 37
- `made` → `mad`: 37
- `good` → `god`: 36
- `her` → `hr`: 34
- `will` → `wil`: 33
- `and` → `in`: 33
- `these` → `thes`: 33

Top deleted words:

- `the`: 187
- `a`: 114
- `and`: 93
- `to`: 60
- `i`: 60
- `it`: 56
- `of`: 52
- `in`: 46
- `you`: 43
- `this`: 42
- `but`: 38
- `is`: 37
- `his`: 36
- `all`: 35
- `with`: 34
- `that`: 30
- `not`: 27
- `was`: 26
- `one`: 25
- `will`: 25
- `there`: 25
- `at`: 25
- `he`: 24
- `for`: 24
- `said`: 23
- `would`: 22
- `or`: 22
- `then`: 21
- `her`: 21
- `great`: 20

Top inserted words:

- `a`: 34
- `un`: 17
- `be`: 16
- `in`: 13
- `som`: 12
- `him`: 9
- `t`: 9
- `an`: 8
- `any`: 8
- `d`: 7
- `re`: 7
- `n`: 7
- `al`: 7
- `every`: 7
- `o`: 6
- `hr`: 5
- `to`: 5
- `ther`: 5
- `her`: 4
- `hous`: 4
- `what`: 4
- `l`: 4
- `with`: 4
- `my`: 4
- `ben`: 4
- `how`: 4
- `man`: 4
- `he`: 4
- `wer`: 3
- `wod`: 3

Duration-bucket WER:

- 10-15s: 0.6262
- <5s: 0.7167
- 5-10s: 0.6433
- >=15s: 0.6340

Highest speaker-level WER:

- 7729: 0.7089
- 1995: 0.7058
- 1188: 0.6960
- 8555: 0.6909
- 4446: 0.6902
- 4507: 0.6833
- 2094: 0.6808
- 260: 0.6800
- 5142: 0.6784
- 2961: 0.6756
- 4077: 0.6752
- 7176: 0.6744
- 3570: 0.6718
- 8455: 0.6642
- 6829: 0.6590
- 3575: 0.6562
- 7021: 0.6561
- 5683: 0.6554
- 4992: 0.6516
- 2300: 0.6498
- 7127: 0.6490
- 8463: 0.6484
- 237: 0.6460
- 8230: 0.6411
- 6930: 0.6409
- 4970: 0.6404
- 2830: 0.6400
- 61: 0.6334
- 1284: 0.6303
- 121: 0.6299

Representative high-error utterances:

- `1089-134691-0018` ref: again again / hyp: a gn agn
- `4446-2271-0002` ref: it's tremendously well put on too / hyp: it s h mn as they wl pur on to
- `8463-294825-0000` ref: it's almost beyond conjecture / hyp: its ms t be nd concuer
- `8555-284449-0014` ref: the former boolooroo groaned / hyp: he f r bdele r grn
- `237-134493-0007` ref: alexandra lets you sleep late / hyp: d agxsngre l as us lep lad

## e15b_mlp_layer8_ctc

Top substitutions:

- `and` → `an`: 48
- `all` → `al`: 34
- `are` → `ar`: 32
- `one` → `on`: 31
- `see` → `se`: 31
- `been` → `ben`: 31
- `made` → `mad`: 27
- `three` → `thre`: 27
- `the` → `he`: 27
- `good` → `god`: 26
- `door` → `dor`: 23
- `will` → `wil`: 23
- `too` → `to`: 22
- `these` → `thes`: 21
- `into` → `to`: 21
- `there` → `ther`: 21
- `know` → `now`: 21
- `while` → `whil`: 21
- `here` → `her`: 21
- `before` → `befor`: 20
- `seen` → `sen`: 20
- `the` → `te`: 20
- `the` → `th`: 19
- `off` → `of`: 18
- `in` → `and`: 18
- `eyes` → `eys`: 17
- `soon` → `son`: 17
- `fire` → `fir`: 16
- `in` → `ind`: 16
- `were` → `wer`: 16

Top deleted words:

- `a`: 51
- `the`: 39
- `and`: 38
- `in`: 33
- `to`: 24
- `i`: 24
- `it`: 20
- `is`: 18
- `there`: 14
- `this`: 14
- `had`: 11
- `with`: 10
- `would`: 10
- `not`: 10
- `no`: 9
- `you`: 9
- `been`: 8
- `they`: 8
- `we`: 8
- `was`: 7
- `some`: 7
- `were`: 7
- `what`: 7
- `lady`: 7
- `he`: 7
- `as`: 7
- `his`: 6
- `oh`: 6
- `have`: 6
- `she`: 6

Top inserted words:

- `a`: 43
- `un`: 25
- `in`: 22
- `al`: 12
- `over`: 9
- `any`: 9
- `be`: 8
- `an`: 7
- `my`: 7
- `i`: 6
- `with`: 6
- `the`: 5
- `t`: 5
- `for`: 5
- `e`: 4
- `te`: 4
- `en`: 4
- `it`: 4
- `ad`: 4
- `he`: 3
- `th`: 3
- `grand`: 3
- `to`: 3
- `de`: 3
- `every`: 3
- `is`: 3
- `of`: 3
- `what`: 3
- `there`: 3
- `r`: 2

Duration-bucket WER:

- 5-10s: 0.3460
- <5s: 0.4597
- 10-15s: 0.3394

Highest speaker-level WER:

- 1272: 0.4318
- 777: 0.4311
- 5694: 0.4166
- 6241: 0.4131
- 652: 0.4125
- 6313: 0.4118
- 1988: 0.4051
- 7850: 0.4047
- 2086: 0.3966
- 5338: 0.3933
- 1919: 0.3922
- 3000: 0.3884
- 1673: 0.3859
- 8842: 0.3849
- 251: 0.3842
- 5895: 0.3826
- 174: 0.3812
- 5536: 0.3805
- 3081: 0.3796
- 2803: 0.3770
- 2412: 0.3766
- 3853: 0.3749
- 84: 0.3613
- 7976: 0.3599
- 2902: 0.3560
- 2428: 0.3516
- 3752: 0.3491
- 6295: 0.3479
- 6319: 0.3472
- 422: 0.3471

Representative high-error utterances:

- `1919-142785-0048` ref: french forcemeat / hyp: frh fors mt
- `2078-142845-0044` ref: italian rusks / hyp: at taian russ
- `8842-304647-0007` ref: most wonderful / hyp: m somnd fl
- `2428-83699-0041` ref: i'm mister christopher from london / hyp: i mestu crisar f r fom lntin
- `1462-170145-0021` ref: alexander flushed angrily / hyp: li sndor flset angrly

## e15b_mlp_layer8_ctc

Top substitutions:

- `and` → `an`: 56
- `all` → `al`: 49
- `been` → `ben`: 38
- `too` → `to`: 33
- `see` → `se`: 30
- `the` → `he`: 29
- `will` → `wil`: 29
- `one` → `on`: 29
- `and` → `in`: 28
- `know` → `now`: 27
- `are` → `ar`: 27
- `made` → `mad`: 26
- `the` → `te`: 26
- `room` → `rom`: 26
- `more` → `mor`: 25
- `tree` → `tre`: 24
- `thee` → `the`: 24
- `while` → `whil`: 23
- `three` → `thre`: 22
- `like` → `lik`: 21
- `these` → `thes`: 21
- `good` → `god`: 20
- `quite` → `quit`: 20
- `before` → `befor`: 20
- `there` → `ther`: 20
- `the` → `th`: 19
- `door` → `dor`: 19
- `in` → `ind`: 18
- `indeed` → `inded`: 17
- `between` → `betwen`: 17

Top deleted words:

- `a`: 56
- `the`: 55
- `and`: 45
- `in`: 35
- `i`: 28
- `it`: 27
- `of`: 26
- `this`: 22
- `to`: 20
- `one`: 19
- `no`: 18
- `is`: 17
- `that`: 15
- `you`: 14
- `then`: 14
- `are`: 14
- `but`: 12
- `not`: 12
- `he`: 11
- `for`: 11
- `or`: 11
- `more`: 11
- `some`: 11
- `all`: 10
- `there`: 10
- `be`: 10
- `him`: 10
- `was`: 10
- `so`: 9
- `her`: 9

Top inserted words:

- `un`: 38
- `a`: 33
- `in`: 30
- `al`: 10
- `an`: 10
- `with`: 9
- `every`: 9
- `be`: 8
- `any`: 8
- `some`: 7
- `t`: 7
- `to`: 7
- `there`: 7
- `the`: 6
- `what`: 6
- `he`: 5
- `i`: 5
- `de`: 5
- `over`: 5
- `under`: 4
- `fir`: 4
- `s`: 4
- `for`: 4
- `mean`: 4
- `where`: 3
- `day`: 3
- `up`: 3
- `ther`: 3
- `re`: 3
- `out`: 3

Duration-bucket WER:

- 10-15s: 0.3437
- <5s: 0.4733
- 5-10s: 0.3708
- >=15s: 0.3594

Highest speaker-level WER:

- 7729: 0.4883
- 1995: 0.4679
- 4077: 0.4375
- 3570: 0.4335
- 2961: 0.4334
- 8555: 0.4324
- 7176: 0.4307
- 2300: 0.4264
- 4507: 0.4219
- 8463: 0.4202
- 1188: 0.4167
- 3575: 0.4119
- 8455: 0.4004
- 260: 0.3983
- 4992: 0.3967
- 2094: 0.3915
- 5142: 0.3886
- 4970: 0.3865
- 6829: 0.3862
- 5683: 0.3857
- 4446: 0.3837
- 121: 0.3754
- 908: 0.3751
- 8230: 0.3735
- 7127: 0.3676
- 1284: 0.3609
- 2830: 0.3608
- 237: 0.3568
- 7021: 0.3473
- 6930: 0.3460

Representative high-error utterances:

- `1089-134691-0024` ref: stephanos dedalos / hyp: ffoin nos dd os
- `260-123286-0020` ref: tuesday august eighteenth / hyp: tuse day agust eight tenth
- `8463-294825-0000` ref: it's almost beyond conjecture / hyp: its eost to be ond conjectur
- `8555-284449-0014` ref: the former boolooroo groaned / hyp: he sfon r thle rd grd
- `237-134493-0007` ref: alexandra lets you sleep late / hyp: agengt rl at s yu lep lade

## e15c_bilstm_layer8_ctc

Top substitutions:

- `and` → `an`: 27
- `too` → `to`: 26
- `in` → `and`: 23
- `and` → `in`: 21
- `in` → `an`: 14
- `a` → `the`: 12
- `in` → `ind`: 11
- `will` → `wil`: 11
- `all` → `al`: 11
- `the` → `a`: 10
- `poor` → `por`: 10
- `been` → `ben`: 10
- `see` → `se`: 10
- `are` → `ar`: 9
- `there` → `ther`: 9
- `and` → `ad`: 9
- `off` → `of`: 9
- `four` → `for`: 9
- `they` → `the`: 8
- `blood` → `blod`: 8
- `add` → `ad`: 8
- `full` → `ful`: 8
- `good` → `god`: 8
- `middle` → `midle`: 7
- `feeling` → `feling`: 7
- `little` → `litle`: 7
- `into` → `to`: 7
- `eggs` → `egs`: 7
- `the` → `th`: 7
- `their` → `ther`: 7

Top deleted words:

- `a`: 43
- `the`: 22
- `and`: 21
- `i`: 15
- `in`: 13
- `to`: 12
- `any`: 6
- `you`: 6
- `were`: 5
- `of`: 5
- `do`: 5
- `see`: 5
- `is`: 5
- `he`: 5
- `it`: 4
- `did`: 4
- `every`: 4
- `not`: 4
- `we`: 4
- `that`: 4
- `are`: 4
- `an`: 4
- `one`: 4
- `his`: 4
- `had`: 3
- `they`: 3
- `don't`: 3
- `us`: 3
- `more`: 3
- `those`: 3

Top inserted words:

- `a`: 30
- `the`: 13
- `in`: 12
- `t`: 8
- `s`: 7
- `any`: 7
- `he`: 6
- `grand`: 6
- `to`: 5
- `al`: 5
- `ad`: 4
- `of`: 4
- `o`: 4
- `i`: 4
- `e`: 3
- `n`: 3
- `an`: 3
- `ind`: 3
- `and`: 3
- `is`: 3
- `sweet`: 3
- `no`: 3
- `mag`: 3
- `dis`: 2
- `nont`: 2
- `tu`: 2
- `ned`: 2
- `at`: 2
- `up`: 2
- `fire`: 2

Duration-bucket WER:

- 5-10s: 0.2061
- <5s: 0.3414
- 10-15s: 0.1967

Highest speaker-level WER:

- 777: 0.3228
- 652: 0.3005
- 5694: 0.2949
- 1272: 0.2905
- 6313: 0.2804
- 6241: 0.2796
- 1919: 0.2700
- 1673: 0.2700
- 1988: 0.2621
- 251: 0.2602
- 5536: 0.2523
- 5895: 0.2507
- 2078: 0.2448
- 7850: 0.2446
- 8842: 0.2431
- 5338: 0.2397
- 3853: 0.2390
- 2086: 0.2383
- 2412: 0.2325
- 84: 0.2320
- 3752: 0.2314
- 1462: 0.2301
- 3081: 0.2278
- 174: 0.2225
- 2035: 0.2188
- 2803: 0.2178
- 2428: 0.2156
- 7976: 0.2149
- 2902: 0.2077
- 6319: 0.2037

Representative high-error utterances:

- `1919-142785-0048` ref: french forcemeat / hyp: fre ntes forts meihth
- `251-136532-0022` ref: lectures / hyp: ic crse
- `3081-166546-0008` ref: george / hyp: tal c
- `2078-142845-0009` ref: illustration italian millet / hyp: il strtion at taigeian molat
- `2078-142845-0048` ref: seventeen thirty four / hyp: s in teeen theirty foor

## e15c_bilstm_layer8_ctc

Top substitutions:

- `and` → `an`: 41
- `too` → `to`: 37
- `in` → `and`: 30
- `and` → `in`: 29
- `thee` → `the`: 24
- `all` → `al`: 21
- `been` → `ben`: 16
- `are` → `ar`: 14
- `the` → `a`: 14
- `soon` → `son`: 13
- `in` → `an`: 13
- `an` → `and`: 13
- `will` → `wil`: 12
- `their` → `ther`: 11
- `and` → `ind`: 11
- `they` → `the`: 11
- `is` → `as`: 9
- `off` → `of`: 9
- `this` → `the`: 9
- `with` → `wit`: 9
- `a` → `the`: 9
- `consumption` → `consemption`: 9
- `good` → `god`: 8
- `poor` → `por`: 8
- `two` → `to`: 8
- `the` → `te`: 8
- `tree` → `tre`: 8
- `robin` → `roban`: 7
- `o` → `oh`: 7
- `see` → `se`: 7

Top deleted words:

- `a`: 45
- `and`: 20
- `the`: 19
- `in`: 17
- `it`: 12
- `you`: 12
- `i`: 11
- `he`: 10
- `to`: 8
- `of`: 8
- `more`: 8
- `this`: 7
- `is`: 7
- `do`: 7
- `all`: 6
- `but`: 5
- `one`: 5
- `be`: 5
- `then`: 4
- `his`: 4
- `not`: 4
- `at`: 4
- `no`: 4
- `some`: 4
- `she`: 4
- `or`: 4
- `last`: 4
- `why`: 4
- `latter`: 4
- `now`: 3

Top inserted words:

- `a`: 24
- `the`: 13
- `in`: 11
- `t`: 11
- `s`: 10
- `to`: 9
- `o`: 8
- `e`: 7
- `some`: 7
- `he`: 7
- `every`: 6
- `i`: 6
- `an`: 6
- `never`: 6
- `up`: 5
- `mont`: 5
- `be`: 4
- `and`: 4
- `house`: 4
- `is`: 4
- `any`: 4
- `un`: 4
- `with`: 4
- `grand`: 4
- `on`: 4
- `can`: 4
- `de`: 3
- `day`: 3
- `there`: 3
- `ten`: 3

Duration-bucket WER:

- 10-15s: 0.2022
- <5s: 0.3573
- 5-10s: 0.2282
- >=15s: 0.1973

Highest speaker-level WER:

- 1995: 0.3513
- 8555: 0.3143
- 3570: 0.3133
- 7176: 0.3031
- 2961: 0.2760
- 8463: 0.2758
- 2300: 0.2741
- 4077: 0.2724
- 7729: 0.2719
- 4992: 0.2719
- 4507: 0.2719
- 3575: 0.2658
- 4446: 0.2641
- 5142: 0.2545
- 1188: 0.2531
- 8455: 0.2464
- 908: 0.2443
- 61: 0.2438
- 260: 0.2410
- 2830: 0.2403
- 6829: 0.2397
- 2094: 0.2349
- 4970: 0.2315
- 121: 0.2242
- 237: 0.2194
- 1089: 0.2165
- 1284: 0.2155
- 6930: 0.2090
- 5105: 0.2080
- 7127: 0.2063

Representative high-error utterances:

- `1089-134691-0024` ref: stephanos dedalos / hyp: s dafer nose deat los
- `3575-170457-0016` ref: farewell madam / hyp: ferm b e am
- `8555-292519-0002` ref: venice / hyp: e es
- `8463-294825-0000` ref: it's almost beyond conjecture / hyp: it som os to be ont conjectoure
- `5105-28241-0010` ref: ocean reigned supreme / hyp: m shin rain tstor pram

## e15d_bilstm_layer8_ctc

Top substitutions:

- `and` → `an`: 39
- `and` → `in`: 23
- `in` → `and`: 19
- `in` → `an`: 16
- `a` → `the`: 10
- `the` → `a`: 10
- `in` → `ind`: 10
- `is` → `as`: 7
- `add` → `ad`: 7
- `bread` → `bred`: 7
- `randal` → `randle`: 7
- `the` → `he`: 6
- `of` → `o`: 6
- `this` → `the`: 6
- `to` → `o`: 6
- `mary` → `marry`: 5
- `o` → `oh`: 5
- `eggs` → `egs`: 5
- `you're` → `your`: 5
- `it's` → `its`: 5
- `one` → `on`: 5
- `and` → `ad`: 5
- `on` → `an`: 5
- `carrie` → `carry`: 5
- `will` → `wil`: 5
- `two` → `to`: 5
- `milner` → `millner`: 5
- `a` → `at`: 5
- `are` → `ar`: 5
- `shaggy` → `shagy`: 4

Top deleted words:

- `a`: 37
- `in`: 16
- `to`: 15
- `and`: 12
- `i`: 11
- `the`: 10
- `it`: 6
- `do`: 5
- `don`: 5
- `this`: 4
- `very`: 4
- `he`: 4
- `you`: 4
- `of`: 3
- `with`: 3
- `white`: 3
- `every`: 3
- `was`: 3
- `dead`: 3
- `any`: 3
- `each`: 3
- `shall`: 3
- `is`: 3
- `that`: 3
- `an`: 3
- `no`: 2
- `mac`: 2
- `own`: 2
- `such`: 2
- `serve`: 2

Top inserted words:

- `a`: 40
- `in`: 14
- `the`: 13
- `i`: 9
- `any`: 8
- `he`: 6
- `and`: 5
- `of`: 5
- `grand`: 5
- `all`: 5
- `some`: 4
- `at`: 4
- `an`: 3
- `up`: 3
- `do`: 3
- `ad`: 3
- `s`: 3
- `te`: 3
- `with`: 3
- `to`: 3
- `be`: 3
- `mag`: 3
- `bus`: 2
- `thi`: 2
- `fire`: 2
- `man`: 2
- `over`: 2
- `ar`: 2
- `can`: 2
- `down`: 2

Duration-bucket WER:

- 5-10s: 0.1545
- <5s: 0.2855
- 10-15s: 0.1467

Highest speaker-level WER:

- 777: 0.2611
- 652: 0.2476
- 1272: 0.2408
- 6313: 0.2234
- 1673: 0.2178
- 5694: 0.2166
- 1919: 0.2146
- 251: 0.2097
- 1988: 0.2058
- 6241: 0.2001
- 5895: 0.1993
- 3752: 0.1935
- 8842: 0.1903
- 7850: 0.1897
- 5536: 0.1894
- 3853: 0.1844
- 2803: 0.1840
- 2902: 0.1832
- 2035: 0.1812
- 5338: 0.1810
- 84: 0.1803
- 2078: 0.1792
- 3081: 0.1769
- 2086: 0.1754
- 2412: 0.1743
- 2428: 0.1713
- 422: 0.1647
- 1462: 0.1638
- 6295: 0.1632
- 6319: 0.1632

Representative high-error utterances:

- `8842-304647-0007` ref: most wonderful / hyp: mos o ln erfl
- `2078-142845-0009` ref: illustration italian millet / hyp: illstration at tige en melit
- `2078-142845-0048` ref: seventeen thirty four / hyp: s in teen thirty f war
- `2428-83699-0041` ref: i'm mister christopher from london / hyp: i mes a cres arf fer fom lndtein
- `1919-142785-0048` ref: french forcemeat / hyp: fringt foris meat

## e15d_bilstm_layer8_ctc

Top substitutions:

- `and` → `an`: 45
- `and` → `in`: 33
- `in` → `and`: 26
- `in` → `an`: 19
- `paul` → `pall`: 13
- `an` → `and`: 12
- `the` → `a`: 10
- `a` → `the`: 10
- `this` → `the`: 9
- `i` → `y`: 9
- `the` → `he`: 9
- `and` → `ind`: 9
- `is` → `as`: 8
- `too` → `to`: 8
- `consumption` → `consemption`: 8
- `thee` → `the`: 8
- `their` → `ther`: 7
- `o` → `oh`: 7
- `shone` → `shown`: 7
- `oh` → `o`: 7
- `is` → `his`: 7
- `as` → `a`: 7
- `ruth` → `roth`: 7
- `robin` → `robbin`: 7
- `kenneth` → `kenith`: 7
- `of` → `o`: 6
- `wholly` → `holy`: 6
- `you` → `ou`: 6
- `of` → `a`: 6
- `i'm` → `im`: 6

Top deleted words:

- `a`: 44
- `in`: 16
- `he`: 11
- `i`: 10
- `the`: 9
- `and`: 9
- `to`: 9
- `is`: 8
- `more`: 7
- `this`: 6
- `it`: 6
- `have`: 5
- `all`: 5
- `why`: 5
- `be`: 5
- `of`: 5
- `at`: 4
- `not`: 4
- `come`: 4
- `some`: 4
- `we`: 4
- `or`: 4
- `an`: 4
- `do`: 4
- `on`: 4
- `if`: 3
- `then`: 3
- `school`: 3
- `was`: 3
- `no`: 3

Top inserted words:

- `a`: 40
- `the`: 15
- `i`: 14
- `in`: 11
- `an`: 11
- `to`: 9
- `s`: 9
- `some`: 9
- `every`: 7
- `up`: 6
- `it`: 6
- `with`: 6
- `never`: 6
- `be`: 5
- `o`: 5
- `there`: 5
- `any`: 5
- `r`: 5
- `he`: 5
- `and`: 5
- `fire`: 4
- `t`: 4
- `e`: 4
- `for`: 4
- `can`: 4
- `un`: 3
- `house`: 3
- `de`: 3
- `at`: 3
- `is`: 3

Duration-bucket WER:

- 10-15s: 0.1581
- <5s: 0.3108
- 5-10s: 0.1859
- >=15s: 0.1525

Highest speaker-level WER:

- 1995: 0.2926
- 8555: 0.2637
- 7176: 0.2594
- 3570: 0.2593
- 2961: 0.2517
- 4507: 0.2448
- 8463: 0.2411
- 7729: 0.2346
- 4992: 0.2296
- 2300: 0.2169
- 4077: 0.2122
- 4446: 0.2118
- 5142: 0.2072
- 260: 0.2058
- 3575: 0.2034
- 908: 0.1995
- 6829: 0.1965
- 8455: 0.1955
- 1188: 0.1937
- 2094: 0.1894
- 4970: 0.1873
- 1284: 0.1872
- 61: 0.1837
- 237: 0.1813
- 2830: 0.1811
- 1089: 0.1804
- 7127: 0.1771
- 5105: 0.1742
- 121: 0.1726
- 8224: 0.1681

Representative high-error utterances:

- `1089-134691-0003` ref: the university / hyp: they youn av vercity
- `3575-170457-0016` ref: farewell madam / hyp: fee bao me dem
- `8555-292519-0002` ref: venice / hyp: e a
- `3729-6852-0010` ref: i never had any family / hyp: gy ne hr h e henye ov he tey
- `1995-1836-0011` ref: positively heroic added cresswell avoiding his sister's eyes / hyp: os it ei ee hor al adit cors waw wortiyg he cis thirs ais

## e16a_top4_finetune

Top substitutions:

- `in` → `and`: 27
- `and` → `in`: 16
- `and` → `an`: 14
- `in` → `ind`: 11
- `until` → `untill`: 9
- `mary` → `marry`: 9
- `a` → `the`: 9
- `randal` → `randle`: 9
- `in` → `an`: 8
- `the` → `he`: 8
- `bartley` → `bartly`: 8
- `it's` → `its`: 8
- `add` → `ad`: 8
- `eggs` → `egs`: 8
- `o` → `oh`: 7
- `the` → `a`: 7
- `guide` → `gide`: 7
- `the` → `te`: 6
- `and` → `ind`: 6
- `serve` → `served`: 6
- `guard` → `gard`: 5
- `key` → `ky`: 5
- `cosette` → `cosete`: 5
- `century` → `sentury`: 5
- `would` → `ould`: 5
- `flour` → `flower`: 5
- `phoebe` → `feeby`: 5
- `through` → `throgh`: 5
- `soul` → `sol`: 5
- `divided` → `devided`: 5

Top deleted words:

- `a`: 21
- `the`: 9
- `in`: 8
- `to`: 6
- `and`: 6
- `that`: 5
- `don`: 5
- `it`: 4
- `any`: 4
- `mary`: 4
- `are`: 4
- `with`: 4
- `of`: 3
- `had`: 3
- `new`: 3
- `all`: 3
- `but`: 3
- `i`: 3
- `state`: 3
- `vita`: 3
- `there`: 2
- `mac`: 2
- `t`: 2
- `up`: 2
- `since`: 2
- `her`: 2
- `ann`: 2
- `dead`: 2
- `on`: 2
- `every`: 2

Top inserted words:

- `a`: 32
- `in`: 9
- `the`: 8
- `of`: 5
- `grand`: 5
- `he`: 5
- `any`: 5
- `sweet`: 5
- `i`: 4
- `over`: 4
- `out`: 4
- `at`: 3
- `to`: 3
- `fore`: 3
- `it`: 3
- `chest`: 3
- `may`: 3
- `news`: 3
- `every`: 3
- `table`: 3
- `mag`: 3
- `after`: 2
- `up`: 2
- `fire`: 2
- `al`: 2
- `some`: 2
- `can`: 2
- `down`: 2
- `foot`: 2
- `where`: 2

Duration-bucket WER:

- 5-10s: 0.1364
- <5s: 0.1470
- 10-15s: 0.1359

Highest speaker-level WER:

- 652: 0.2121
- 777: 0.1944
- 6313: 0.1834
- 1272: 0.1711
- 1919: 0.1704
- 1673: 0.1622
- 3752: 0.1603
- 1988: 0.1600
- 251: 0.1592
- 5694: 0.1589
- 5895: 0.1532
- 6241: 0.1498
- 8842: 0.1497
- 3000: 0.1463
- 3576: 0.1400
- 5338: 0.1399
- 2902: 0.1396
- 7850: 0.1394
- 3081: 0.1381
- 84: 0.1371
- 6295: 0.1316
- 2086: 0.1316
- 2078: 0.1312
- 2803: 0.1309
- 5536: 0.1290
- 1462: 0.1256
- 8297: 0.1256
- 174: 0.1225
- 1993: 0.1223
- 2412: 0.1208

Representative high-error utterances:

- `2428-83705-0036` ref: someone sniggered / hyp: some one snigred
- `174-50561-0018` ref: bed time children / hyp: bead timed childron
- `1919-142785-0039` ref: illustration marjoram / hyp: illoustration mardgorim
- `1919-142785-0046` ref: illustration basil / hyp: illustrationbasil
- `1919-142785-0048` ref: french forcemeat / hyp: french fource meet

## e16a_top4_finetune

Top substitutions:

- `and` → `in`: 27
- `in` → `and`: 18
- `christ` → `crist`: 16
- `a` → `the`: 12
- `and` → `an`: 12
- `an` → `and`: 11
- `until` → `untill`: 9
- `in` → `an`: 8
- `the` → `a`: 8
- `in` → `ind`: 8
- `paul` → `pal`: 8
- `will` → `wil`: 7
- `leisure` → `leasure`: 7
- `o` → `oh`: 7
- `bear` → `beare`: 7
- `paul` → `pall`: 7
- `ruth` → `routh`: 7
- `robin` → `robbin`: 6
- `wholly` → `holy`: 6
- `blue` → `ble`: 6
- `anyone` → `one`: 6
- `tree` → `tre`: 6
- `the` → `th`: 6
- `edison` → `edicon`: 6
- `roughly` → `ruffly`: 6
- `where` → `were`: 6
- `thee` → `the`: 6
- `a` → `ha`: 6
- `too` → `to`: 5
- `christ` → `criste`: 5

Top deleted words:

- `a`: 21
- `and`: 8
- `the`: 7
- `or`: 6
- `i`: 5
- `it`: 5
- `in`: 4
- `through`: 4
- `new`: 4
- `this`: 4
- `to`: 4
- `do`: 4
- `la`: 4
- `school`: 3
- `have`: 3
- `will`: 3
- `you`: 3
- `latter`: 3
- `are`: 3
- `ben`: 3
- `mademoiselle`: 3
- `all`: 2
- `corn`: 2
- `that`: 2
- `he`: 2
- `more`: 2
- `with`: 2
- `of`: 2
- `miss`: 2
- `an`: 2

Top inserted words:

- `a`: 19
- `the`: 14
- `any`: 13
- `in`: 11
- `i`: 7
- `up`: 6
- `some`: 6
- `every`: 6
- `he`: 6
- `there`: 5
- `house`: 5
- `with`: 5
- `to`: 5
- `under`: 5
- `where`: 4
- `fire`: 4
- `all`: 4
- `be`: 3
- `patch`: 3
- `forth`: 3
- `for`: 3
- `as`: 3
- `so`: 3
- `top`: 3
- `other`: 3
- `ten`: 2
- `what`: 2
- `more`: 2
- `belling`: 2
- `out`: 2

Duration-bucket WER:

- 10-15s: 0.1403
- <5s: 0.1609
- 5-10s: 0.1409
- >=15s: 0.1383

Highest speaker-level WER:

- 3570: 0.2019
- 7729: 0.1937
- 8555: 0.1887
- 2961: 0.1860
- 7176: 0.1843
- 4992: 0.1835
- 4077: 0.1767
- 1995: 0.1761
- 2300: 0.1694
- 8463: 0.1694
- 260: 0.1659
- 2830: 0.1631
- 61: 0.1614
- 6829: 0.1573
- 1089: 0.1572
- 4970: 0.1513
- 2094: 0.1506
- 4507: 0.1458
- 5142: 0.1449
- 908: 0.1391
- 5105: 0.1389
- 4446: 0.1386
- 3575: 0.1332
- 1284: 0.1327
- 7127: 0.1312
- 1188: 0.1312
- 6930: 0.1300
- 8224: 0.1251
- 121: 0.1237
- 5639: 0.1228

Representative high-error utterances:

- `1089-134691-0024` ref: stephanos dedalos / hyp: stephenose dead los
- `3575-170457-0016` ref: farewell madam / hyp: faire well madame
- `1089-134691-0018` ref: again again / hyp: againagain
- `237-134500-0025` ref: oh emil / hyp: o aimal
- `260-123288-0014` ref: hans stirs not / hyp: haondsstears notied

## e16b_top5_finetune

Top substitutions:

- `in` → `and`: 25
- `until` → `untill`: 16
- `and` → `in`: 15
- `in` → `ind`: 12
- `and` → `an`: 12
- `a` → `the`: 10
- `in` → `an`: 9
- `the` → `a`: 9
- `add` → `ad`: 9
- `eggs` → `egs`: 8
- `mary` → `marry`: 8
- `and` → `ind`: 8
- `o` → `oh`: 7
- `phoebe` → `feeby`: 7
- `randal` → `randle`: 7
- `pepper` → `peper`: 6
- `alexander` → `allexander`: 5
- `it's` → `its`: 5
- `century` → `sentury`: 5
- `ingredients` → `engredience`: 5
- `occurred` → `occured`: 5
- `eye` → `ey`: 5
- `i'm` → `im`: 5
- `serve` → `served`: 5
- `luggage` → `lugage`: 4
- `i'm` → `iam`: 4
- `prayers` → `prairs`: 4
- `italian` → `etalian`: 4
- `there` → `thre`: 4
- `in` → `i`: 4

Top deleted words:

- `a`: 22
- `and`: 9
- `the`: 9
- `to`: 8
- `in`: 8
- `don`: 5
- `any`: 4
- `of`: 3
- `it`: 3
- `will`: 3
- `new`: 3
- `you're`: 3
- `i`: 3
- `they`: 3
- `for`: 3
- `mary`: 3
- `dead`: 3
- `that`: 3
- `you`: 2
- `mac`: 2
- `salt`: 2
- `when`: 2
- `since`: 2
- `had`: 2
- `every`: 2
- `with`: 2
- `asked`: 2
- `an`: 2
- `fair`: 2
- `are`: 2

Top inserted words:

- `a`: 30
- `in`: 9
- `the`: 8
- `any`: 8
- `i`: 5
- `sweet`: 5
- `of`: 4
- `grand`: 4
- `he`: 4
- `e`: 4
- `fire`: 3
- `it`: 3
- `and`: 3
- `table`: 3
- `mag`: 3
- `up`: 2
- `arch`: 2
- `down`: 2
- `garro`: 2
- `degaro`: 2
- `with`: 2
- `chest`: 2
- `who`: 2
- `some`: 2
- `hand`: 2
- `over`: 2
- `near`: 2
- `s`: 2
- `man`: 2
- `at`: 2

Duration-bucket WER:

- 5-10s: 0.1178
- <5s: 0.1341
- 10-15s: 0.1185

Highest speaker-level WER:

- 652: 0.2082
- 777: 0.1686
- 1272: 0.1632
- 1919: 0.1571
- 6313: 0.1571
- 1673: 0.1564
- 251: 0.1506
- 3576: 0.1461
- 8842: 0.1427
- 3752: 0.1359
- 5694: 0.1352
- 5895: 0.1302
- 84: 0.1301
- 2078: 0.1264
- 2086: 0.1258
- 6241: 0.1242
- 7850: 0.1241
- 1988: 0.1230
- 2902: 0.1169
- 5338: 0.1145
- 2803: 0.1140
- 6295: 0.1137
- 2428: 0.1127
- 5536: 0.1125
- 3081: 0.1096
- 2035: 0.1094
- 3000: 0.1072
- 2412: 0.1064
- 1462: 0.1061
- 174: 0.1037

Representative high-error utterances:

- `2428-83705-0036` ref: someone sniggered / hyp: some one sniggared
- `8842-304647-0001` ref: quinci impara a stupirti / hyp: quen che empara s to beerti
- `1919-142785-0039` ref: illustration marjoram / hyp: illoustration mardorim
- `1919-142785-0048` ref: french forcemeat / hyp: french forse meat
- `2078-142845-0041` ref: illustration buns / hyp: ellustration bunns

## e16b_top5_finetune

Top substitutions:

- `and` → `in`: 29
- `in` → `and`: 22
- `christ` → `crist`: 17
- `a` → `the`: 14
- `an` → `and`: 13
- `and` → `an`: 12
- `in` → `an`: 10
- `until` → `untill`: 10
- `the` → `a`: 9
- `paul` → `pall`: 9
- `thee` → `the`: 8
- `o` → `oh`: 8
- `uncas` → `uncus`: 7
- `the` → `th`: 7
- `robin` → `robbin`: 7
- `will` → `wil`: 6
- `system` → `sistom`: 6
- `eye` → `ey`: 6
- `paul` → `pal`: 6
- `servadac` → `servadack`: 6
- `knife` → `nife`: 6
- `wholly` → `holy`: 5
- `and` → `ind`: 5
- `sought` → `saught`: 5
- `anyone` → `one`: 5
- `this` → `the`: 5
- `holmes` → `homes`: 5
- `it's` → `its`: 5
- `gentleman` → `gentlemen`: 5
- `is` → `his`: 5

Top deleted words:

- `a`: 20
- `in`: 12
- `the`: 10
- `or`: 8
- `it`: 6
- `you`: 5
- `to`: 4
- `and`: 3
- `school`: 3
- `will`: 3
- `this`: 3
- `on`: 3
- `latter`: 3
- `of`: 3
- `la`: 3
- `new`: 3
- `mademoiselle`: 3
- `all`: 2
- `through`: 2
- `for`: 2
- `looked`: 2
- `he`: 2
- `more`: 2
- `could`: 2
- `i`: 2
- `would`: 2
- `do`: 2
- `ben`: 2
- `e`: 2
- `any`: 2

Top inserted words:

- `a`: 17
- `in`: 10
- `the`: 10
- `any`: 9
- `i`: 9
- `some`: 8
- `every`: 7
- `to`: 6
- `he`: 6
- `up`: 5
- `there`: 4
- `house`: 4
- `an`: 4
- `mount`: 4
- `mont`: 4
- `be`: 3
- `under`: 3
- `what`: 3
- `fire`: 3
- `with`: 3
- `forth`: 3
- `for`: 3
- `un`: 3
- `as`: 3
- `here`: 3
- `grand`: 3
- `top`: 3
- `fits`: 3
- `play`: 3
- `never`: 3

Duration-bucket WER:

- 10-15s: 0.1216
- <5s: 0.1423
- 5-10s: 0.1225
- >=15s: 0.1229

Highest speaker-level WER:

- 3570: 0.1810
- 8555: 0.1686
- 1995: 0.1674
- 7729: 0.1668
- 7176: 0.1666
- 2961: 0.1600
- 4992: 0.1575
- 61: 0.1573
- 4077: 0.1543
- 8463: 0.1532
- 260: 0.1471
- 2300: 0.1424
- 1089: 0.1387
- 2830: 0.1349
- 5105: 0.1343
- 908: 0.1317
- 2094: 0.1305
- 4507: 0.1302
- 6829: 0.1290
- 5142: 0.1287
- 4970: 0.1236
- 237: 0.1144
- 3575: 0.1132
- 1188: 0.1127
- 7127: 0.1115
- 6930: 0.1115
- 5639: 0.1105
- 4446: 0.1098
- 5683: 0.1095
- 1284: 0.1077

Representative high-error utterances:

- `3575-170457-0016` ref: farewell madam / hyp: faire well madame
- `61-70968-0038` ref: robin fitzooth / hyp: robein fits outh
- `260-123288-0014` ref: hans stirs not / hyp: han's stear is notid
- `1089-134691-0018` ref: again again / hyp: againagain
- `1089-134691-0024` ref: stephanos dedalos / hyp: stefhinos dedlos

## e16c_top8_finetune

Top substitutions:

- `in` → `and`: 31
- `and` → `in`: 19
- `and` → `an`: 17
- `the` → `a`: 9
- `a` → `the`: 9
- `and` → `ind`: 9
- `in` → `an`: 8
- `until` → `untill`: 8
- `alexander` → `allexander`: 7
- `o` → `oh`: 7
- `randal` → `randl`: 7
- `asleep` → `sleep`: 6
- `in` → `ind`: 6
- `flour` → `flower`: 6
- `mary` → `marry`: 6
- `add` → `ad`: 5
- `carrie` → `carry`: 5
- `wrote` → `rote`: 5
- `occurred` → `occured`: 5
- `this` → `the`: 5
- `guide` → `gide`: 5
- `macklewain` → `macklewaine`: 5
- `kaliko` → `calico`: 4
- `harry` → `herry`: 4
- `to` → `too`: 4
- `week` → `weak`: 4
- `pepper` → `peper`: 4
- `eggs` → `egs`: 4
- `eye` → `ey`: 4
- `divine` → `devine`: 4

Top deleted words:

- `a`: 27
- `in`: 10
- `the`: 10
- `to`: 9
- `and`: 8
- `don`: 5
- `of`: 4
- `it`: 4
- `i`: 4
- `there`: 3
- `for`: 3
- `mary`: 3
- `any`: 3
- `an`: 3
- `one`: 3
- `no`: 2
- `you're`: 2
- `some`: 2
- `me`: 2
- `new`: 2
- `red`: 2
- `on`: 2
- `every`: 2
- `that`: 2
- `are`: 2
- `grub`: 2
- `is`: 2
- `ice`: 2
- `elm`: 2
- `our`: 2

Top inserted words:

- `a`: 29
- `any`: 10
- `the`: 8
- `in`: 6
- `i`: 4
- `some`: 4
- `sweet`: 4
- `to`: 3
- `fire`: 3
- `grand`: 3
- `he`: 3
- `every`: 3
- `with`: 3
- `and`: 3
- `an`: 3
- `there`: 3
- `out`: 3
- `mounta`: 3
- `mag`: 3
- `main`: 2
- `up`: 2
- `arch`: 2
- `shon`: 2
- `corn`: 2
- `of`: 2
- `o`: 2
- `it`: 2
- `t`: 2
- `garro`: 2
- `foot`: 2

Duration-bucket WER:

- 5-10s: 0.1120
- <5s: 0.1269
- 10-15s: 0.1137

Highest speaker-level WER:

- 652: 0.2011
- 1272: 0.1692
- 777: 0.1571
- 6313: 0.1558
- 8842: 0.1393
- 1673: 0.1367
- 1919: 0.1345
- 3576: 0.1339
- 251: 0.1287
- 7850: 0.1277
- 3752: 0.1272
- 5694: 0.1265
- 2078: 0.1248
- 5895: 0.1240
- 2086: 0.1239
- 2803: 0.1219
- 1988: 0.1198
- 84: 0.1176
- 6241: 0.1171
- 3000: 0.1135
- 2902: 0.1117
- 5536: 0.1108
- 2035: 0.1103
- 1462: 0.1092
- 5338: 0.1057
- 174: 0.1050
- 8297: 0.1033
- 1993: 0.1022
- 3853: 0.0996
- 2428: 0.0984

Representative high-error utterances:

- `1919-142785-0048` ref: french forcemeat / hyp: ffrench force meat
- `1272-135031-0016` ref: kaliko hesitated / hyp: caligo hesiptated
- `1919-142785-0056` ref: fried bread crumbs / hyp: frie bred croums
- `2078-142845-0000` ref: kirkleatham yeast / hyp: cerkeoley tham yeast
- `2078-142845-0044` ref: italian rusks / hyp: etalion rusxks

## e16c_top8_finetune

Top substitutions:

- `and` → `in`: 28
- `and` → `an`: 17
- `christ` → `crist`: 15
- `in` → `and`: 15
- `a` → `the`: 14
- `an` → `and`: 12
- `the` → `a`: 11
- `in` → `an`: 10
- `until` → `untill`: 9
- `and` → `ind`: 8
- `paul` → `pall`: 8
- `o` → `oh`: 7
- `shone` → `shown`: 7
- `this` → `the`: 7
- `system` → `sistom`: 7
- `paul` → `pal`: 7
- `hester` → `hesther`: 6
- `anyone` → `one`: 6
- `less` → `orless`: 6
- `bartley` → `bartly`: 6
- `robin` → `roben`: 6
- `is` → `as`: 5
- `sought` → `saught`: 5
- `uncas` → `uncus`: 5
- `man` → `men`: 5
- `holmes` → `homes`: 5
- `edison` → `edicon`: 5
- `the` → `te`: 5
- `silvia` → `sylvia`: 5
- `slang` → `slaing`: 5

Top deleted words:

- `a`: 29
- `and`: 13
- `the`: 8
- `to`: 7
- `or`: 6
- `in`: 5
- `have`: 5
- `on`: 5
- `new`: 4
- `an`: 3
- `some`: 3
- `of`: 3
- `this`: 3
- `he`: 3
- `i`: 3
- `more`: 3
- `latter`: 3
- `ben`: 3
- `mademoiselle`: 3
- `then`: 2
- `school`: 2
- `free`: 2
- `all`: 2
- `other`: 2
- `will`: 2
- `can`: 2
- `there`: 2
- `who`: 2
- `for`: 2
- `we`: 2

Top inserted words:

- `a`: 14
- `in`: 13
- `any`: 11
- `the`: 11
- `i`: 9
- `up`: 6
- `every`: 6
- `never`: 6
- `to`: 5
- `he`: 5
- `for`: 4
- `an`: 4
- `all`: 4
- `mount`: 4
- `fits`: 4
- `where`: 3
- `be`: 3
- `there`: 3
- `fire`: 3
- `with`: 3
- `as`: 3
- `forth`: 3
- `it`: 3
- `main`: 3
- `is`: 3
- `else`: 3
- `top`: 3
- `play`: 3
- `brittan`: 3
- `after`: 2

Duration-bucket WER:

- 10-15s: 0.1183
- <5s: 0.1310
- 5-10s: 0.1152
- >=15s: 0.1176

Highest speaker-level WER:

- 3570: 0.1695
- 2961: 0.1678
- 7176: 0.1652
- 8555: 0.1605
- 7729: 0.1520
- 260: 0.1502
- 4992: 0.1486
- 1995: 0.1455
- 8463: 0.1444
- 2300: 0.1440
- 4077: 0.1412
- 1089: 0.1331
- 61: 0.1323
- 4507: 0.1292
- 2830: 0.1277
- 6829: 0.1242
- 908: 0.1235
- 5142: 0.1228
- 2094: 0.1216
- 5105: 0.1174
- 4970: 0.1124
- 3575: 0.1096
- 6930: 0.1091
- 1284: 0.1064
- 4446: 0.1052
- 7127: 0.1051
- 5639: 0.1050
- 8455: 0.1039
- 121: 0.1023
- 1320: 0.0991

Representative high-error utterances:

- `3575-170457-0016` ref: farewell madam / hyp: fair well madame
- `61-70968-0038` ref: robin fitzooth / hyp: roppein fits outh
- `1089-134691-0024` ref: stephanos dedalos / hyp: stuffonous tdeadlos
- `121-123852-0001` ref: ay me / hyp: i mee
- `260-123288-0014` ref: hans stirs not / hyp: handstirs night

## e17_k1000_centroid_bilstm_ctc

Top substitutions:

- `and` → `an`: 39
- `in` → `and`: 31
- `and` → `in`: 24
- `are` → `ar`: 17
- `off` → `of`: 15
- `in` → `ind`: 15
- `the` → `a`: 14
- `in` → `an`: 14
- `a` → `the`: 11
- `into` → `to`: 11
- `too` → `two`: 10
- `four` → `for`: 10
- `foot` → `fot`: 9
- `i'm` → `i`: 8
- `have` → `had`: 8
- `this` → `the`: 8
- `you're` → `your`: 7
- `two` → `too`: 7
- `too` → `to`: 7
- `the` → `th`: 7
- `add` → `ad`: 7
- `there` → `the`: 7
- `has` → `is`: 7
- `where` → `were`: 7
- `an` → `and`: 7
- `there` → `ther`: 7
- `manner` → `maner`: 6
- `i'm` → `im`: 6
- `and` → `ind`: 6
- `and` → `ad`: 6

Top deleted words:

- `a`: 64
- `the`: 30
- `in`: 20
- `i`: 19
- `and`: 12
- `of`: 11
- `to`: 11
- `it`: 9
- `all`: 8
- `had`: 8
- `you`: 7
- `go`: 7
- `there`: 7
- `are`: 7
- `no`: 6
- `do`: 6
- `this`: 6
- `for`: 6
- `was`: 6
- `any`: 6
- `we`: 6
- `an`: 6
- `did`: 5
- `us`: 5
- `one`: 5
- `mary`: 5
- `he`: 5
- `they`: 5
- `some`: 5
- `see`: 5

Top inserted words:

- `a`: 60
- `the`: 25
- `in`: 21
- `an`: 13
- `i`: 9
- `and`: 9
- `for`: 8
- `he`: 7
- `o`: 7
- `with`: 6
- `any`: 6
- `be`: 6
- `to`: 5
- `t`: 5
- `there`: 5
- `over`: 5
- `grand`: 4
- `fire`: 3
- `down`: 3
- `has`: 3
- `te`: 3
- `s`: 3
- `at`: 3
- `news`: 3
- `is`: 3
- `on`: 3
- `all`: 3
- `every`: 3
- `e`: 3
- `miss`: 3

Duration-bucket WER:

- 5-10s: 0.2851
- <5s: 0.4528
- 10-15s: 0.2701

Highest speaker-level WER:

- 777: 0.4304
- 1272: 0.3980
- 1988: 0.3931
- 652: 0.3912
- 6313: 0.3742
- 5694: 0.3684
- 1673: 0.3476
- 5338: 0.3474
- 5895: 0.3454
- 251: 0.3403
- 5536: 0.3366
- 6241: 0.3357
- 7850: 0.3336
- 3000: 0.3329
- 2803: 0.3307
- 2078: 0.3296
- 1919: 0.3265
- 3752: 0.3215
- 2902: 0.3159
- 3853: 0.3143
- 3081: 0.3123
- 1462: 0.3112
- 2412: 0.3094
- 2086: 0.3041
- 2035: 0.3040
- 8842: 0.3036
- 84: 0.3025
- 174: 0.2950
- 6319: 0.2946
- 6295: 0.2915

Representative high-error utterances:

- `2078-142845-0044` ref: italian rusks / hyp: a t an resx
- `251-136532-0022` ref: lectures / hyp: wekxe erse
- `777-126732-0081` ref: comfortable dear / hyp: ou ea e tar
- `1462-170138-0003` ref: alexander exclaimed mildly / hyp: a was anvy exclamed madly
- `1462-170145-0021` ref: alexander flushed angrily / hyp: wi sen e fast angriley

## e17_k1000_centroid_bilstm_ctc

Top substitutions:

- `and` → `an`: 47
- `in` → `and`: 38
- `and` → `in`: 35
- `in` → `an`: 32
- `an` → `and`: 23
- `are` → `ar`: 18
- `the` → `a`: 15
- `a` → `the`: 15
- `is` → `as`: 14
- `thee` → `the`: 14
- `where` → `were`: 14
- `too` → `to`: 11
- `it` → `i`: 11
- `off` → `of`: 10
- `the` → `th`: 10
- `gave` → `give`: 10
- `in` → `ind`: 10
- `within` → `in`: 10
- `and` → `ind`: 9
- `of` → `a`: 9
- `will` → `wil`: 8
- `blue` → `blu`: 8
- `tree` → `tre`: 8
- `faith` → `fath`: 7
- `red` → `read`: 7
- `think` → `thank`: 7
- `are` → `or`: 7
- `this` → `the`: 7
- `an` → `in`: 7
- `as` → `is`: 7

Top deleted words:

- `a`: 86
- `the`: 23
- `i`: 21
- `in`: 21
- `and`: 15
- `do`: 12
- `it`: 11
- `of`: 11
- `you`: 10
- `no`: 10
- `to`: 10
- `this`: 9
- `new`: 9
- `he`: 8
- `at`: 8
- `is`: 8
- `or`: 8
- `fir`: 8
- `then`: 7
- `there`: 7
- `for`: 7
- `some`: 7
- `are`: 7
- `but`: 6
- `her`: 6
- `what`: 6
- `that`: 5
- `more`: 5
- `any`: 5
- `so`: 5

Top inserted words:

- `a`: 67
- `the`: 21
- `i`: 19
- `in`: 19
- `o`: 12
- `with`: 11
- `and`: 10
- `an`: 9
- `t`: 9
- `some`: 8
- `any`: 8
- `what`: 6
- `there`: 6
- `e`: 6
- `he`: 6
- `be`: 5
- `we`: 5
- `ar`: 5
- `y`: 5
- `is`: 5
- `they`: 5
- `other`: 5
- `ha`: 4
- `so`: 4
- `s`: 4
- `fire`: 4
- `out`: 4
- `on`: 4
- `to`: 4
- `at`: 4

Duration-bucket WER:

- 10-15s: 0.2851
- <5s: 0.4698
- 5-10s: 0.3227
- >=15s: 0.2744

Highest speaker-level WER:

- 3570: 0.4261
- 1995: 0.4171
- 8555: 0.4101
- 8463: 0.4016
- 7176: 0.4014
- 7729: 0.3997
- 2300: 0.3756
- 4507: 0.3740
- 2961: 0.3616
- 2830: 0.3564
- 260: 0.3552
- 4077: 0.3534
- 4992: 0.3492
- 4446: 0.3477
- 3575: 0.3453
- 2094: 0.3430
- 6829: 0.3423
- 8455: 0.3365
- 4970: 0.3356
- 5142: 0.3269
- 61: 0.3255
- 1221: 0.3234
- 908: 0.3220
- 121: 0.3194
- 1284: 0.3172
- 1188: 0.3148
- 5105: 0.3116
- 7127: 0.3004
- 237: 0.2993
- 3729: 0.2939

Representative high-error utterances:

- `8555-292519-0002` ref: venice / hyp: i we o
- `1089-134691-0003` ref: the university / hyp: thith u a fosit
- `3729-6852-0010` ref: i never had any family / hyp: be ne hen he ais a rhinis ire be
- `121-127105-0021` ref: won't you tell douglas / hyp: n s ein toe o e s
- `8463-294825-0000` ref: it's almost beyond conjecture / hyp: it sam mlose to bean kin ecter

## e17_k500_centroid_bilstm_ctc

Top substitutions:

- `and` → `an`: 42
- `in` → `and`: 25
- `and` → `in`: 24
- `are` → `ar`: 18
- `into` → `to`: 17
- `in` → `an`: 16
- `a` → `the`: 15
- `an` → `and`: 13
- `in` → `ind`: 13
- `is` → `as`: 12
- `of` → `o`: 12
- `on` → `an`: 11
- `full` → `ful`: 11
- `to` → `o`: 10
- `been` → `ben`: 10
- `her` → `er`: 10
- `off` → `of`: 10
- `how` → `ho`: 9
- `there` → `they`: 9
- `their` → `ther`: 9
- `all` → `al`: 8
- `the` → `a`: 8
- `too` → `to`: 8
- `it` → `at`: 8
- `three` → `thre`: 8
- `too` → `two`: 8
- `a` → `o`: 8
- `seemed` → `seeme`: 8
- `there` → `the`: 8
- `there` → `their`: 8

Top deleted words:

- `a`: 99
- `in`: 26
- `the`: 26
- `i`: 15
- `it`: 15
- `you`: 15
- `is`: 14
- `to`: 14
- `and`: 13
- `be`: 12
- `some`: 11
- `how`: 11
- `do`: 10
- `no`: 9
- `are`: 9
- `he`: 8
- `for`: 8
- `what`: 7
- `an`: 7
- `there`: 7
- `was`: 7
- `that`: 6
- `put`: 6
- `this`: 6
- `they`: 6
- `of`: 6
- `any`: 6
- `room`: 5
- `these`: 5
- `good`: 5

Top inserted words:

- `a`: 77
- `in`: 19
- `and`: 14
- `the`: 14
- `to`: 12
- `i`: 11
- `for`: 11
- `over`: 9
- `an`: 7
- `be`: 7
- `un`: 7
- `o`: 7
- `any`: 6
- `with`: 5
- `he`: 5
- `s`: 5
- `e`: 5
- `no`: 5
- `is`: 5
- `man`: 5
- `t`: 4
- `grand`: 4
- `can`: 4
- `all`: 4
- `y`: 4
- `some`: 3
- `up`: 3
- `men`: 3
- `they`: 3
- `you`: 3

Duration-bucket WER:

- 5-10s: 0.3352
- <5s: 0.5001
- 10-15s: 0.3138

Highest speaker-level WER:

- 777: 0.4656
- 652: 0.4456
- 1272: 0.4299
- 6313: 0.4255
- 1988: 0.4140
- 5694: 0.3992
- 6241: 0.3989
- 1673: 0.3975
- 5338: 0.3963
- 251: 0.3908
- 5536: 0.3863
- 3853: 0.3861
- 2078: 0.3856
- 8842: 0.3841
- 1919: 0.3840
- 3000: 0.3834
- 5895: 0.3809
- 2902: 0.3735
- 3752: 0.3720
- 7850: 0.3687
- 1462: 0.3674
- 2803: 0.3612
- 2412: 0.3608
- 2086: 0.3546
- 84: 0.3534
- 6319: 0.3516
- 2035: 0.3426
- 174: 0.3425
- 7976: 0.3375
- 422: 0.3353

Representative high-error utterances:

- `1462-170145-0021` ref: alexander flushed angrily / hyp: he an tea fush d angrlly
- `1919-142785-0024` ref: illustration ginger / hyp: a whe tration denger
- `1919-142785-0039` ref: illustration marjoram / hyp: in whecationm man trom
- `2078-142845-0009` ref: illustration italian millet / hyp: anstation ait tian me l et
- `2078-142845-0044` ref: italian rusks / hyp: i tdey an rasx

## e17_k500_centroid_bilstm_ctc

Top substitutions:

- `and` → `in`: 45
- `and` → `an`: 43
- `in` → `and`: 35
- `an` → `and`: 27
- `in` → `an`: 19
- `are` → `ar`: 17
- `is` → `as`: 16
- `too` → `to`: 15
- `oh` → `o`: 15
- `men` → `man`: 15
- `their` → `ther`: 14
- `thee` → `the`: 14
- `the` → `a`: 14
- `there` → `ther`: 14
- `too` → `two`: 13
- `a` → `the`: 12
- `here` → `her`: 11
- `of` → `a`: 11
- `into` → `to`: 10
- `what` → `wat`: 10
- `off` → `of`: 10
- `where` → `were`: 10
- `blue` → `ble`: 9
- `it` → `at`: 9
- `been` → `ben`: 8
- `that` → `the`: 8
- `less` → `les`: 8
- `within` → `in`: 8
- `it's` → `its`: 8
- `your` → `you`: 7

Top deleted words:

- `a`: 124
- `it`: 23
- `the`: 22
- `to`: 20
- `in`: 20
- `i`: 20
- `you`: 17
- `be`: 17
- `is`: 15
- `can`: 15
- `and`: 14
- `all`: 14
- `or`: 13
- `her`: 11
- `do`: 10
- `if`: 9
- `am`: 9
- `there`: 8
- `for`: 8
- `of`: 8
- `some`: 8
- `at`: 7
- `so`: 7
- `what`: 7
- `will`: 7
- `then`: 6
- `he`: 6
- `that`: 6
- `but`: 6
- `come`: 6

Top inserted words:

- `a`: 95
- `in`: 31
- `i`: 21
- `and`: 15
- `the`: 15
- `he`: 13
- `an`: 13
- `o`: 13
- `some`: 12
- `be`: 11
- `with`: 10
- `to`: 8
- `any`: 8
- `as`: 8
- `there`: 7
- `un`: 7
- `man`: 7
- `on`: 6
- `her`: 5
- `ae`: 5
- `e`: 5
- `it`: 5
- `over`: 5
- `where`: 4
- `you`: 4
- `ar`: 4
- `every`: 4
- `ba`: 4
- `en`: 4
- `ho`: 4

Duration-bucket WER:

- 10-15s: 0.3349
- <5s: 0.5120
- 5-10s: 0.3682
- >=15s: 0.3280

Highest speaker-level WER:

- 3570: 0.4767
- 7176: 0.4594
- 4507: 0.4573
- 7729: 0.4544
- 1995: 0.4484
- 8463: 0.4323
- 4077: 0.4313
- 2300: 0.4280
- 8555: 0.4257
- 2961: 0.4187
- 3575: 0.4162
- 2830: 0.4113
- 4446: 0.3974
- 260: 0.3951
- 121: 0.3852
- 4970: 0.3843
- 4992: 0.3834
- 8455: 0.3815
- 5142: 0.3808
- 2094: 0.3803
- 6829: 0.3741
- 61: 0.3707
- 1188: 0.3704
- 237: 0.3691
- 1284: 0.3670
- 908: 0.3623
- 8224: 0.3548
- 5683: 0.3478
- 5105: 0.3469
- 6930: 0.3398

Representative high-error utterances:

- `237-134500-0001` ref: marie sighed / hyp: whet we o sie
- `8463-294825-0000` ref: it's almost beyond conjecture / hyp: its sa os to be on comecur
- `260-123286-0010` ref: sunday august sixteenth / hyp: sinday ab gu sixt tave
- `5105-28240-0002` ref: exclaimed servadac keeping his eye unmoved at his telescope / hyp: ink edy so ve drai ti he as a omn o ee hish oats co
- `5105-28241-0014` ref: another circumstance was most remarkable / hyp: an thti socen se intristo ash a marckable

## e18_k200_embedding_bilstm_ctc

Top substitutions:

- `and` → `an`: 58
- `and` → `in`: 53
- `in` → `an`: 44
- `in` → `and`: 38
- `at` → `it`: 36
- `been` → `ben`: 32
- `into` → `to`: 29
- `are` → `ar`: 23
- `all` → `al`: 23
- `in` → `ind`: 23
- `the` → `a`: 22
- `it` → `at`: 22
- `a` → `the`: 18
- `and` → `ind`: 16
- `their` → `ther`: 15
- `it` → `i`: 15
- `here` → `her`: 15
- `off` → `of`: 13
- `there` → `their`: 12
- `four` → `for`: 12
- `one` → `on`: 11
- `of` → `o`: 11
- `a` → `e`: 10
- `made` → `mad`: 10
- `than` → `then`: 10
- `our` → `ar`: 10
- `dead` → `ded`: 10
- `this` → `the`: 9
- `an` → `and`: 9
- `too` → `two`: 9

Top deleted words:

- `a`: 138
- `in`: 50
- `the`: 50
- `i`: 34
- `he`: 27
- `it`: 26
- `and`: 26
- `you`: 21
- `all`: 17
- `to`: 16
- `but`: 16
- `for`: 15
- `there`: 14
- `is`: 14
- `an`: 13
- `some`: 12
- `this`: 12
- `of`: 12
- `her`: 11
- `old`: 11
- `how`: 11
- `what`: 11
- `are`: 10
- `up`: 10
- `see`: 10
- `been`: 9
- `our`: 9
- `at`: 9
- `had`: 9
- `we`: 8

Top inserted words:

- `a`: 77
- `in`: 39
- `o`: 17
- `for`: 12
- `and`: 12
- `e`: 11
- `i`: 11
- `the`: 11
- `an`: 9
- `at`: 8
- `he`: 8
- `t`: 8
- `to`: 7
- `with`: 7
- `is`: 7
- `ad`: 7
- `en`: 6
- `ind`: 6
- `al`: 6
- `on`: 6
- `be`: 6
- `ther`: 5
- `any`: 5
- `yo`: 4
- `are`: 4
- `con`: 4
- `there`: 4
- `it`: 4
- `some`: 4
- `or`: 3

Duration-bucket WER:

- 5-10s: 0.4571
- <5s: 0.6082
- 10-15s: 0.4419

Highest speaker-level WER:

- 1272: 0.5721
- 777: 0.5717
- 6313: 0.5438
- 1988: 0.5346
- 652: 0.5276
- 5338: 0.5264
- 3000: 0.5183
- 5694: 0.5162
- 2803: 0.5147
- 422: 0.5137
- 6241: 0.5046
- 7850: 0.5045
- 2078: 0.5040
- 251: 0.5033
- 3853: 0.5004
- 1919: 0.4979
- 2412: 0.4943
- 5536: 0.4938
- 3752: 0.4937
- 2086: 0.4890
- 1462: 0.4852
- 84: 0.4835
- 7976: 0.4818
- 1673: 0.4797
- 5895: 0.4792
- 2902: 0.4782
- 8842: 0.4775
- 3576: 0.4642
- 3081: 0.4633
- 1993: 0.4632

Representative high-error utterances:

- `8297-275155-0021` ref: honestly / hyp: l whe spleay
- `1462-170145-0021` ref: alexander flushed angrily / hyp: i rs an etf flshed engraly
- `2078-142845-0009` ref: illustration italian millet / hyp: istra tion i cart r ialit
- `5694-64025-0000` ref: shiloh / hyp: har m
- `8842-304647-0007` ref: most wonderful / hyp: rs hor ee fu

## e18_k200_embedding_bilstm_ctc

Top substitutions:

- `and` → `in`: 84
- `and` → `an`: 57
- `in` → `an`: 55
- `in` → `and`: 50
- `the` → `a`: 34
- `all` → `al`: 29
- `are` → `ar`: 27
- `it` → `at`: 24
- `an` → `and`: 22
- `at` → `it`: 21
- `an` → `in`: 20
- `into` → `to`: 18
- `in` → `ind`: 18
- `a` → `the`: 18
- `been` → `ben`: 17
- `own` → `on`: 15
- `is` → `as`: 15
- `thee` → `the`: 15
- `i` → `y`: 14
- `the` → `tha`: 14
- `see` → `se`: 14
- `there` → `their`: 14
- `and` → `ind`: 13
- `will` → `wil`: 13
- `too` → `to`: 13
- `the` → `he`: 13
- `her` → `er`: 12
- `the` → `e`: 12
- `this` → `the`: 12
- `there` → `ther`: 12

Top deleted words:

- `a`: 158
- `the`: 61
- `in`: 43
- `it`: 32
- `and`: 31
- `i`: 31
- `you`: 23
- `but`: 21
- `is`: 21
- `for`: 21
- `to`: 19
- `of`: 18
- `all`: 17
- `he`: 16
- `have`: 15
- `an`: 15
- `there`: 14
- `no`: 14
- `her`: 13
- `be`: 12
- `are`: 12
- `will`: 12
- `come`: 12
- `do`: 12
- `at`: 12
- `what`: 11
- `that`: 11
- `am`: 11
- `some`: 11
- `this`: 11

Top inserted words:

- `a`: 119
- `in`: 46
- `an`: 26
- `i`: 21
- `and`: 18
- `the`: 17
- `he`: 17
- `ar`: 12
- `some`: 12
- `t`: 12
- `be`: 11
- `o`: 11
- `e`: 11
- `to`: 10
- `at`: 9
- `you`: 9
- `con`: 8
- `s`: 8
- `it`: 7
- `any`: 7
- `al`: 6
- `there`: 5
- `as`: 5
- `her`: 5
- `ye`: 5
- `for`: 5
- `n`: 5
- `on`: 5
- `yu`: 5
- `with`: 5

Duration-bucket WER:

- 10-15s: 0.4524
- <5s: 0.6324
- 5-10s: 0.4904
- >=15s: 0.4455

Highest speaker-level WER:

- 3570: 0.5928
- 7729: 0.5691
- 8463: 0.5621
- 1995: 0.5595
- 2300: 0.5516
- 4507: 0.5448
- 3575: 0.5437
- 4446: 0.5418
- 8555: 0.5379
- 5142: 0.5341
- 7176: 0.5317
- 4992: 0.5282
- 4077: 0.5247
- 260: 0.5227
- 2830: 0.5152
- 2094: 0.5123
- 2961: 0.5121
- 6829: 0.5118
- 8455: 0.5022
- 8224: 0.4995
- 4970: 0.4966
- 1188: 0.4946
- 237: 0.4914
- 5683: 0.4895
- 61: 0.4787
- 121: 0.4742
- 908: 0.4712
- 6930: 0.4683
- 5105: 0.4635
- 8230: 0.4608

Representative high-error utterances:

- `2094-142345-0016` ref: spinning indeed / hyp: ss ad e ied
- `8555-292519-0002` ref: venice / hyp: anh a
- `5105-28241-0014` ref: another circumstance was most remarkable / hyp: a is hik he hnt hois teo thor mokble
- `672-122797-0011` ref: and then what happens then / hyp: s one frra wath r dands lee in
- `3729-6852-0036` ref: when the king comes to paris everybody calls out vive le roi / hyp: wenhi toee ooe sho p haerse and heeied ay hour as eodb fe in thor rhr h ah way

## e20a_1h_layer8_bilstm_ctc

Top substitutions:

- `and` → `an`: 42
- `in` → `an`: 25
- `their` → `ther`: 23
- `and` → `in`: 22
- `soul` → `sol`: 19
- `the` → `he`: 18
- `with` → `wit`: 17
- `are` → `ar`: 17
- `in` → `and`: 16
- `will` → `wil`: 15
- `asked` → `ased`: 15
- `four` → `for`: 14
- `oh` → `o`: 12
- `half` → `haf`: 12
- `two` → `to`: 12
- `you` → `yo`: 12
- `good` → `god`: 11
- `one` → `wone`: 11
- `looking` → `loking`: 11
- `a` → `the`: 10
- `look` → `lok`: 10
- `three` → `thre`: 10
- `been` → `ben`: 10
- `off` → `of`: 10
- `another` → `nother`: 10
- `in` → `ind`: 10
- `asleep` → `sleep`: 9
- `all` → `al`: 9
- `keep` → `kee`: 9
- `death` → `deth`: 9

Top deleted words:

- `a`: 36
- `the`: 17
- `to`: 16
- `in`: 13
- `i`: 9
- `each`: 8
- `do`: 6
- `and`: 6
- `it`: 5
- `such`: 5
- `with`: 5
- `had`: 5
- `not`: 5
- `don`: 5
- `is`: 4
- `no`: 4
- `of`: 4
- `you`: 4
- `been`: 4
- `old`: 4
- `were`: 4
- `mary`: 4
- `an`: 4
- `we`: 4
- `that`: 4
- `little`: 3
- `this`: 3
- `are`: 3
- `new`: 3
- `good`: 3

Top inserted words:

- `a`: 57
- `the`: 21
- `in`: 15
- `an`: 13
- `some`: 12
- `i`: 11
- `s`: 11
- `my`: 9
- `grand`: 8
- `can`: 8
- `any`: 8
- `for`: 7
- `ar`: 7
- `he`: 6
- `and`: 6
- `be`: 6
- `un`: 5
- `where`: 5
- `out`: 4
- `with`: 4
- `up`: 4
- `on`: 4
- `t`: 4
- `am`: 4
- `all`: 4
- `sweet`: 4
- `to`: 4
- `gin`: 4
- `mag`: 4
- `en`: 3

Duration-bucket WER:

- 5-10s: 0.2742
- <5s: 0.4087
- 10-15s: 0.2762

Highest speaker-level WER:

- 777: 0.3938
- 1673: 0.3847
- 1272: 0.3731
- 6241: 0.3520
- 652: 0.3494
- 6313: 0.3385
- 251: 0.3365
- 1919: 0.3357
- 1988: 0.3344
- 8842: 0.3339
- 5895: 0.3295
- 5694: 0.3281
- 2086: 0.3260
- 5536: 0.3218
- 422: 0.3196
- 3000: 0.3178
- 2412: 0.3170
- 7850: 0.3156
- 5338: 0.3141
- 3752: 0.3049
- 3853: 0.2996
- 2803: 0.2980
- 3081: 0.2934
- 1462: 0.2925
- 2902: 0.2897
- 84: 0.2892
- 8297: 0.2865
- 2078: 0.2864
- 2277: 0.2855
- 2428: 0.2817

Representative high-error utterances:

- `777-126732-0017` ref: very characteristic perfectly typical / hyp: vhar y car thareisti pou ean the ta icle
- `1919-142785-0048` ref: french forcemeat / hyp: frinte for its met
- `2428-83705-0036` ref: someone sniggered / hyp: som i s nigored
- `3081-166546-0021` ref: i emphasised complacently / hyp: y am the fai's t complayescandtly
- `3081-166546-0073` ref: yes / hyp: igart us

## e20a_1h_layer8_bilstm_ctc

Top substitutions:

- `and` → `an`: 57
- `and` → `in`: 36
- `in` → `and`: 29
- `are` → `ar`: 22
- `in` → `an`: 21
- `their` → `ther`: 21
- `oh` → `o`: 21
- `will` → `wil`: 20
- `thee` → `the`: 16
- `with` → `wit`: 15
- `too` → `to`: 15
- `the` → `he`: 15
- `an` → `and`: 15
- `sir` → `ser`: 15
- `is` → `as`: 14
- `something` → `thing`: 14
- `our` → `or`: 13
- `they` → `the`: 13
- `memory` → `memery`: 13
- `i'm` → `im`: 13
- `before` → `befor`: 12
- `feel` → `fel`: 12
- `in` → `ind`: 11
- `two` → `to`: 10
- `sea` → `see`: 10
- `you` → `yo`: 10
- `earth` → `erth`: 10
- `see` → `se`: 10
- `asked` → `ased`: 10
- `bartley` → `bartly`: 10

Top deleted words:

- `a`: 46
- `i`: 21
- `the`: 16
- `in`: 15
- `and`: 13
- `to`: 13
- `of`: 11
- `more`: 10
- `his`: 8
- `you`: 8
- `or`: 7
- `it`: 7
- `with`: 6
- `is`: 6
- `do`: 6
- `all`: 5
- `one`: 5
- `why`: 5
- `but`: 5
- `our`: 5
- `he`: 4
- `can`: 4
- `that`: 4
- `at`: 4
- `so`: 4
- `no`: 4
- `how`: 4
- `each`: 4
- `this`: 4
- `as`: 4

Top inserted words:

- `a`: 40
- `some`: 24
- `the`: 22
- `s`: 21
- `in`: 20
- `an`: 20
- `un`: 16
- `any`: 16
- `re`: 12
- `he`: 9
- `can`: 9
- `be`: 8
- `to`: 8
- `dis`: 8
- `al`: 8
- `o`: 7
- `i`: 7
- `out`: 7
- `every`: 7
- `under`: 6
- `what`: 6
- `is`: 6
- `as`: 6
- `and`: 6
- `there`: 5
- `mis`: 5
- `all`: 5
- `with`: 5
- `my`: 5
- `it`: 5

Duration-bucket WER:

- 10-15s: 0.2772
- <5s: 0.4307
- 5-10s: 0.3035
- >=15s: 0.2796

Highest speaker-level WER:

- 1995: 0.4108
- 8555: 0.3886
- 7729: 0.3875
- 7176: 0.3843
- 4507: 0.3719
- 3570: 0.3714
- 2300: 0.3552
- 8463: 0.3548
- 8455: 0.3517
- 2961: 0.3443
- 4077: 0.3426
- 4446: 0.3373
- 260: 0.3365
- 4992: 0.3321
- 6829: 0.3214
- 1188: 0.3202
- 3575: 0.3188
- 5105: 0.3085
- 8224: 0.3060
- 121: 0.3043
- 5142: 0.3042
- 1089: 0.3031
- 237: 0.3029
- 2094: 0.3028
- 2830: 0.3023
- 908: 0.3001
- 5683: 0.2955
- 1284: 0.2943
- 4970: 0.2929
- 7127: 0.2885

Representative high-error utterances:

- `8555-292519-0002` ref: venice / hyp: s eay is
- `2094-142345-0016` ref: spinning indeed / hyp: s hi liy lyin eiled
- `3575-170457-0016` ref: farewell madam / hyp: fair au m teirmn
- `672-122797-0049` ref: squeak squeak / hyp: s crik s qcurick
- `8555-284447-0016` ref: fine glorious / hyp: fi rih ri ari's

## e20b_3h_layer8_bilstm_ctc

Top substitutions:

- `and` → `an`: 36
- `in` → `and`: 27
- `and` → `in`: 21
- `in` → `an`: 15
- `in` → `ind`: 12
- `the` → `a`: 11
- `four` → `for`: 11
- `of` → `o`: 10
- `a` → `the`: 9
- `mary` → `marry`: 9
- `or` → `ar`: 9
- `soul` → `sol`: 9
- `bartley` → `bartly`: 8
- `into` → `to`: 8
- `bread` → `bred`: 8
- `randal` → `randle`: 8
- `been` → `ben`: 7
- `i'm` → `im`: 7
- `two` → `to`: 7
- `they` → `the`: 6
- `was` → `as`: 6
- `true` → `tru`: 6
- `other` → `eachother`: 6
- `eggs` → `egs`: 6
- `add` → `ad`: 6
- `good` → `god`: 6
- `either` → `ither`: 6
- `here` → `her`: 6
- `enough` → `anough`: 6
- `as` → `is`: 6

Top deleted words:

- `a`: 54
- `the`: 18
- `in`: 17
- `to`: 16
- `i`: 12
- `and`: 8
- `with`: 7
- `it`: 6
- `each`: 6
- `he`: 6
- `this`: 5
- `do`: 5
- `was`: 5
- `but`: 5
- `no`: 4
- `little`: 4
- `all`: 4
- `some`: 4
- `will`: 4
- `they`: 4
- `how`: 4
- `mary`: 4
- `of`: 3
- `good`: 3
- `such`: 3
- `or`: 3
- `you`: 3
- `is`: 3
- `don`: 3
- `young`: 3

Top inserted words:

- `a`: 36
- `in`: 14
- `the`: 13
- `an`: 10
- `s`: 8
- `he`: 7
- `i`: 6
- `any`: 6
- `some`: 5
- `is`: 5
- `grand`: 5
- `al`: 4
- `and`: 4
- `ad`: 3
- `on`: 3
- `for`: 3
- `with`: 3
- `to`: 3
- `be`: 3
- `there`: 3
- `sweet`: 3
- `do`: 3
- `ever`: 3
- `dis`: 3
- `all`: 3
- `mag`: 3
- `af`: 2
- `man`: 2
- `up`: 2
- `e`: 2

Duration-bucket WER:

- 5-10s: 0.1949
- <5s: 0.3239
- 10-15s: 0.1884

Highest speaker-level WER:

- 777: 0.3207
- 1272: 0.2806
- 652: 0.2752
- 5694: 0.2632
- 1988: 0.2613
- 1919: 0.2577
- 1673: 0.2572
- 6313: 0.2528
- 2078: 0.2496
- 5895: 0.2480
- 6241: 0.2463
- 8842: 0.2422
- 251: 0.2402
- 5338: 0.2368
- 5536: 0.2308
- 2086: 0.2278
- 7850: 0.2275
- 3000: 0.2270
- 2803: 0.2235
- 84: 0.2226
- 3853: 0.2216
- 2412: 0.2211
- 3752: 0.2204
- 422: 0.2196
- 2902: 0.2147
- 3081: 0.2114
- 2428: 0.2104
- 1462: 0.2075
- 2035: 0.2027
- 6319: 0.2015

Representative high-error utterances:

- `1919-142785-0048` ref: french forcemeat / hyp: frings for is meat
- `3081-166546-0073` ref: yes / hyp: hit s
- `5694-64025-0000` ref: shiloh / hyp: sihil o
- `8842-304647-0007` ref: most wonderful / hyp: mno sto mlon terfl
- `777-126732-0017` ref: very characteristic perfectly typical / hyp: very car tursti pou ent they ta picl

## e20b_3h_layer8_bilstm_ctc

Top substitutions:

- `in` → `and`: 34
- `and` → `an`: 34
- `and` → `in`: 33
- `in` → `an`: 21
- `an` → `and`: 14
- `thee` → `the`: 14
- `paul` → `pall`: 13
- `are` → `ar`: 10
- `or` → `ar`: 10
- `too` → `to`: 10
- `bartley` → `bartly`: 10
- `two` → `to`: 9
- `and` → `ind`: 9
- `see` → `se`: 9
- `a` → `the`: 9
- `memory` → `memery`: 9
- `of` → `o`: 8
- `this` → `the`: 8
- `the` → `a`: 8
- `is` → `as`: 7
- `christ` → `crist`: 7
- `tongue` → `tong`: 7
- `o` → `oh`: 7
- `oh` → `o`: 7
- `all` → `al`: 7
- `servant` → `servent`: 7
- `breath` → `breth`: 7
- `to` → `o`: 7
- `is` → `his`: 7
- `tree` → `tre`: 7

Top deleted words:

- `a`: 47
- `to`: 18
- `i`: 14
- `in`: 13
- `the`: 11
- `and`: 10
- `he`: 8
- `more`: 8
- `or`: 7
- `all`: 6
- `of`: 5
- `at`: 5
- `it`: 5
- `on`: 5
- `be`: 5
- `you`: 4
- `there`: 4
- `not`: 4
- `come`: 4
- `so`: 4
- `but`: 4
- `an`: 4
- `la`: 4
- `new`: 4
- `our`: 4
- `how`: 3
- `this`: 3
- `school`: 3
- `know`: 3
- `any`: 3

Top inserted words:

- `a`: 32
- `s`: 19
- `in`: 17
- `the`: 15
- `every`: 12
- `an`: 12
- `some`: 11
- `o`: 10
- `he`: 10
- `any`: 10
- `be`: 9
- `i`: 8
- `there`: 6
- `to`: 6
- `house`: 6
- `for`: 6
- `en`: 5
- `n`: 5
- `and`: 5
- `is`: 5
- `where`: 4
- `they`: 4
- `up`: 4
- `what`: 4
- `ar`: 4
- `sun`: 4
- `all`: 4
- `t`: 4
- `grand`: 4
- `it`: 4

Duration-bucket WER:

- 10-15s: 0.1969
- <5s: 0.3549
- 5-10s: 0.2270
- >=15s: 0.1907

Highest speaker-level WER:

- 1995: 0.3333
- 3570: 0.3005
- 7176: 0.2990
- 8555: 0.2949
- 4507: 0.2917
- 8463: 0.2806
- 4992: 0.2727
- 2961: 0.2716
- 7729: 0.2667
- 2300: 0.2635
- 4077: 0.2631
- 4446: 0.2608
- 260: 0.2551
- 3575: 0.2464
- 8455: 0.2449
- 6829: 0.2444
- 908: 0.2360
- 5142: 0.2347
- 2830: 0.2323
- 1188: 0.2299
- 2094: 0.2267
- 61: 0.2221
- 1089: 0.2213
- 5105: 0.2210
- 1284: 0.2202
- 237: 0.2180
- 4970: 0.2180
- 121: 0.2171
- 7127: 0.2150
- 3729: 0.2128

Representative high-error utterances:

- `1089-134691-0003` ref: the university / hyp: they you n evercity
- `1089-134691-0024` ref: stephanos dedalos / hyp: steatf anos deat los
- `3575-170457-0016` ref: farewell madam / hyp: ferm lo med om
- `672-122797-0049` ref: squeak squeak / hyp: s qrik s quick
- `8555-292519-0002` ref: venice / hyp: n s

## wav2vec2_discrete_k100_bilstm_ctc

Top substitutions:

- `and` → `an`: 82
- `and` → `in`: 81
- `in` → `and`: 78
- `in` → `an`: 63
- `as` → `is`: 47
- `his` → `is`: 45
- `is` → `as`: 44
- `in` → `ind`: 40
- `the` → `a`: 35
- `it` → `at`: 31
- `they` → `the`: 29
- `two` → `to`: 25
- `of` → `o`: 25
- `the` → `te`: 24
- `into` → `to`: 23
- `and` → `ind`: 23
- `the` → `tha`: 22
- `been` → `ben`: 22
- `at` → `it`: 22
- `to` → `o`: 21
- `a` → `the`: 21
- `are` → `ar`: 21
- `the` → `he`: 19
- `too` → `to`: 18
- `a` → `at`: 17
- `this` → `the`: 17
- `his` → `as`: 17
- `their` → `ther`: 17
- `all` → `al`: 17
- `to` → `a`: 15

Top deleted words:

- `a`: 141
- `the`: 82
- `and`: 59
- `i`: 52
- `in`: 46
- `to`: 43
- `he`: 38
- `it`: 38
- `of`: 35
- `you`: 26
- `were`: 21
- `we`: 21
- `there`: 19
- `be`: 18
- `was`: 17
- `could`: 17
- `this`: 17
- `they`: 17
- `his`: 16
- `have`: 16
- `are`: 16
- `is`: 15
- `as`: 15
- `all`: 15
- `an`: 14
- `on`: 13
- `but`: 13
- `had`: 12
- `will`: 12
- `that`: 12

Top inserted words:

- `a`: 62
- `i`: 31
- `an`: 31
- `and`: 30
- `the`: 25
- `in`: 25
- `to`: 21
- `o`: 15
- `at`: 12
- `for`: 11
- `of`: 10
- `t`: 9
- `it`: 9
- `is`: 9
- `e`: 9
- `re`: 8
- `en`: 8
- `no`: 8
- `s`: 7
- `were`: 7
- `as`: 7
- `he`: 7
- `ind`: 7
- `was`: 6
- `ar`: 6
- `af`: 6
- `over`: 5
- `her`: 5
- `be`: 5
- `on`: 5

Duration-bucket WER:

- 5-10s: 0.5878
- <5s: 0.7222
- 10-15s: 0.5643

Highest speaker-level WER:

- 777: 0.7080
- 1272: 0.6886
- 5694: 0.6877
- 5338: 0.6781
- 6313: 0.6771
- 2803: 0.6625
- 2078: 0.6624
- 1988: 0.6495
- 422: 0.6471
- 652: 0.6443
- 6241: 0.6366
- 3752: 0.6280
- 1919: 0.6273
- 2412: 0.6257
- 3853: 0.6242
- 5536: 0.6228
- 7850: 0.6196
- 251: 0.6187
- 1462: 0.6147
- 2277: 0.6126
- 2902: 0.6003
- 8842: 0.6003
- 1993: 0.5998
- 6295: 0.5991
- 1673: 0.5968
- 7976: 0.5955
- 2035: 0.5919
- 3000: 0.5839
- 2086: 0.5834
- 174: 0.5763

Representative high-error utterances:

- `1272-135031-0016` ref: kaliko hesitated / hyp: i ily hiso daded
- `1462-170138-0003` ref: alexander exclaimed mildly / hyp: an was adi exclime t mabely
- `2078-142845-0003` ref: seventeen eighteen / hyp: sevieng to night tam
- `2078-142845-0009` ref: illustration italian millet / hyp: in aisdusthion a carin then heowt
- `5694-64025-0000` ref: shiloh / hyp: soem o

## wav2vec2_discrete_k100_bilstm_ctc

Top substitutions:

- `in` → `and`: 115
- `and` → `an`: 111
- `and` → `in`: 102
- `in` → `an`: 73
- `is` → `as`: 61
- `in` → `ind`: 52
- `the` → `a`: 41
- `the` → `te`: 40
- `and` → `ind`: 38
- `his` → `is`: 38
- `as` → `is`: 35
- `too` → `to`: 31
- `a` → `the`: 30
- `it` → `at`: 29
- `is` → `his`: 29
- `of` → `o`: 26
- `the` → `he`: 24
- `that` → `the`: 23
- `all` → `al`: 22
- `a` → `i`: 21
- `two` → `to`: 20
- `their` → `ther`: 20
- `are` → `ar`: 20
- `the` → `ther`: 19
- `they` → `the`: 19
- `see` → `se`: 19
- `of` → `af`: 19
- `to` → `o`: 19
- `an` → `in`: 18
- `at` → `it`: 18

Top deleted words:

- `a`: 154
- `the`: 81
- `in`: 54
- `to`: 51
- `and`: 49
- `i`: 46
- `you`: 31
- `it`: 30
- `of`: 29
- `he`: 22
- `is`: 20
- `for`: 20
- `on`: 19
- `do`: 18
- `all`: 18
- `have`: 18
- `there`: 18
- `we`: 18
- `if`: 17
- `but`: 17
- `they`: 17
- `as`: 17
- `this`: 16
- `his`: 15
- `be`: 15
- `so`: 15
- `are`: 15
- `will`: 14
- `her`: 14
- `had`: 14

Top inserted words:

- `a`: 103
- `in`: 51
- `the`: 40
- `an`: 37
- `i`: 35
- `and`: 35
- `to`: 26
- `o`: 22
- `e`: 16
- `is`: 16
- `be`: 14
- `t`: 14
- `as`: 14
- `he`: 12
- `for`: 11
- `with`: 11
- `y`: 10
- `te`: 10
- `do`: 10
- `some`: 10
- `of`: 10
- `at`: 10
- `ar`: 9
- `con`: 9
- `ad`: 8
- `we`: 8
- `s`: 8
- `ind`: 7
- `al`: 7
- `on`: 7

Duration-bucket WER:

- 10-15s: 0.5733
- <5s: 0.7294
- 5-10s: 0.6157
- >=15s: 0.5670

Highest speaker-level WER:

- 3570: 0.7022
- 1995: 0.6980
- 8463: 0.6919
- 4446: 0.6752
- 7729: 0.6725
- 5142: 0.6677
- 4992: 0.6664
- 3575: 0.6662
- 8455: 0.6613
- 4507: 0.6583
- 7176: 0.6505
- 260: 0.6448
- 2300: 0.6416
- 8555: 0.6382
- 4077: 0.6350
- 1188: 0.6296
- 61: 0.6280
- 8224: 0.6276
- 237: 0.6245
- 2830: 0.6234
- 2094: 0.6227
- 2961: 0.6142
- 6829: 0.6070
- 4970: 0.6060
- 121: 0.6050
- 5683: 0.5974
- 1284: 0.5912
- 5105: 0.5863
- 7127: 0.5818
- 8230: 0.5812

Representative high-error utterances:

- `8555-292519-0002` ref: venice / hyp: i e e
- `2094-142345-0016` ref: spinning indeed / hyp: sust ma i e id
- `1089-134691-0003` ref: the university / hyp: nca o ther aty
- `8463-294825-0000` ref: it's almost beyond conjecture / hyp: i saoe o st be ind e acure
- `121-127105-0021` ref: won't you tell douglas / hyp: an sand the i o the as

## wav2vec2_discrete_k200_bilstm_ctc

Top substitutions:

- `and` → `in`: 96
- `and` → `an`: 88
- `in` → `and`: 75
- `in` → `an`: 49
- `his` → `is`: 43
- `and` → `ind`: 36
- `it` → `at`: 29
- `is` → `as`: 28
- `in` → `ind`: 28
- `two` → `to`: 26
- `the` → `a`: 25
- `a` → `the`: 24
- `at` → `it`: 23
- `they` → `the`: 23
- `too` → `to`: 22
- `into` → `to`: 22
- `as` → `is`: 22
- `are` → `ar`: 21
- `it` → `a`: 18
- `been` → `ben`: 18
- `an` → `in`: 18
- `all` → `al`: 17
- `to` → `o`: 17
- `the` → `te`: 16
- `this` → `the`: 16
- `to` → `a`: 16
- `off` → `of`: 16
- `a` → `to`: 16
- `is` → `his`: 16
- `three` → `thre`: 15

Top deleted words:

- `a`: 147
- `the`: 48
- `to`: 47
- `in`: 47
- `i`: 46
- `and`: 42
- `you`: 24
- `of`: 22
- `it`: 21
- `he`: 20
- `are`: 17
- `had`: 17
- `with`: 16
- `his`: 16
- `do`: 15
- `all`: 15
- `we`: 15
- `were`: 12
- `there`: 12
- `how`: 12
- `be`: 12
- `but`: 12
- `as`: 11
- `was`: 11
- `on`: 11
- `an`: 10
- `for`: 10
- `then`: 10
- `know`: 10
- `she`: 9

Top inserted words:

- `a`: 68
- `in`: 42
- `an`: 25
- `i`: 24
- `to`: 18
- `o`: 16
- `and`: 16
- `e`: 15
- `is`: 14
- `the`: 14
- `s`: 12
- `at`: 11
- `for`: 10
- `he`: 10
- `al`: 10
- `there`: 10
- `be`: 9
- `it`: 9
- `some`: 8
- `ad`: 8
- `over`: 8
- `on`: 7
- `un`: 7
- `but`: 6
- `u`: 6
- `re`: 6
- `en`: 6
- `no`: 6
- `her`: 6
- `we`: 5

Duration-bucket WER:

- 5-10s: 0.4948
- <5s: 0.6427
- 10-15s: 0.4725

Highest speaker-level WER:

- 777: 0.6428
- 5694: 0.6134
- 1272: 0.5910
- 5338: 0.5812
- 1988: 0.5804
- 6313: 0.5670
- 2078: 0.5600
- 1673: 0.5597
- 652: 0.5536
- 6241: 0.5422
- 422: 0.5412
- 7850: 0.5405
- 2412: 0.5358
- 1462: 0.5296
- 3000: 0.5296
- 2277: 0.5250
- 1919: 0.5236
- 2902: 0.5236
- 3853: 0.5212
- 3752: 0.5205
- 7976: 0.5204
- 5895: 0.5190
- 5536: 0.5178
- 251: 0.5148
- 2803: 0.5124
- 2035: 0.5112
- 2086: 0.5081
- 8842: 0.5052
- 174: 0.5050
- 1993: 0.4928

Representative high-error utterances:

- `1272-135031-0016` ref: kaliko hesitated / hyp: so bur his etated
- `1462-170138-0003` ref: alexander exclaimed mildly / hyp: a was end ne exclaimed to miletly
- `1919-142785-0024` ref: illustration ginger / hyp: o i chration enser
- `2078-142845-0003` ref: seventeen eighteen / hyp: seven to nagt tame
- `2078-142845-0009` ref: illustration italian millet / hyp: lstathion at tee a in nelet

## wav2vec2_discrete_k200_bilstm_ctc

Top substitutions:

- `in` → `and`: 117
- `and` → `in`: 117
- `and` → `an`: 101
- `in` → `an`: 53
- `in` → `ind`: 41
- `is` → `as`: 40
- `and` → `ind`: 40
- `too` → `to`: 29
- `that` → `the`: 27
- `it` → `at`: 26
- `as` → `is`: 26
- `are` → `ar`: 26
- `a` → `the`: 25
- `his` → `is`: 25
- `an` → `and`: 23
- `the` → `a`: 23
- `all` → `al`: 23
- `is` → `his`: 22
- `the` → `he`: 22
- `into` → `to`: 21
- `an` → `in`: 20
- `will` → `wil`: 19
- `thee` → `the`: 19
- `at` → `it`: 17
- `of` → `o`: 17
- `two` → `to`: 17
- `a` → `at`: 17
- `they` → `the`: 16
- `to` → `a`: 16
- `of` → `af`: 16

Top deleted words:

- `a`: 156
- `to`: 57
- `in`: 50
- `and`: 47
- `the`: 45
- `i`: 43
- `of`: 26
- `it`: 24
- `you`: 24
- `but`: 23
- `for`: 20
- `is`: 19
- `at`: 16
- `his`: 15
- `this`: 15
- `be`: 15
- `he`: 14
- `if`: 14
- `had`: 14
- `her`: 14
- `there`: 13
- `an`: 13
- `can`: 12
- `are`: 12
- `so`: 12
- `all`: 11
- `she`: 11
- `or`: 11
- `then`: 10
- `will`: 10

Top inserted words:

- `a`: 92
- `in`: 53
- `an`: 33
- `i`: 31
- `to`: 24
- `o`: 23
- `the`: 23
- `and`: 19
- `as`: 15
- `on`: 14
- `he`: 14
- `ind`: 14
- `e`: 12
- `is`: 12
- `it`: 11
- `at`: 10
- `al`: 10
- `we`: 10
- `f`: 9
- `some`: 9
- `en`: 9
- `for`: 8
- `her`: 7
- `mis`: 7
- `t`: 7
- `all`: 7
- `be`: 7
- `man`: 7
- `over`: 7
- `ad`: 7

Duration-bucket WER:

- 10-15s: 0.4803
- <5s: 0.6547
- 5-10s: 0.5213
- >=15s: 0.4716

Highest speaker-level WER:

- 3570: 0.6232
- 8463: 0.6161
- 7729: 0.6038
- 1995: 0.5939
- 4446: 0.5850
- 5142: 0.5796
- 8555: 0.5773
- 3575: 0.5745
- 8455: 0.5734
- 4992: 0.5713
- 2300: 0.5712
- 260: 0.5626
- 4507: 0.5563
- 7176: 0.5536
- 2094: 0.5488
- 1188: 0.5417
- 2961: 0.5407
- 2830: 0.5361
- 6829: 0.5294
- 4077: 0.5278
- 61: 0.5246
- 237: 0.5245
- 8224: 0.5230
- 4970: 0.5086
- 1284: 0.5037
- 5683: 0.5024
- 121: 0.4947
- 1580: 0.4893
- 8230: 0.4891
- 7127: 0.4830

Representative high-error utterances:

- `1089-134691-0003` ref: the university / hyp: at s eve veo ity
- `2094-142345-0022` ref: mister ottley's indeed / hyp: mes sth out lat in deed
- `237-134500-0001` ref: marie sighed / hyp: my tend he sied
- `8555-292519-0002` ref: venice / hyp: an ose
- `5105-28241-0014` ref: another circumstance was most remarkable / hyp: an th sok en hintrars om mus har macuble

## wav2vec2_discrete_k50_bilstm_ctc

Top substitutions:

- `and` → `an`: 72
- `and` → `in`: 64
- `a` → `the`: 56
- `in` → `and`: 56
- `in` → `an`: 47
- `his` → `is`: 47
- `the` → `te`: 46
- `the` → `a`: 38
- `the` → `he`: 37
- `is` → `as`: 36
- `all` → `al`: 35
- `the` → `to`: 28
- `as` → `is`: 28
- `to` → `the`: 27
- `in` → `ind`: 26
- `the` → `tha`: 25
- `of` → `o`: 24
- `two` → `to`: 24
- `into` → `to`: 23
- `that` → `the`: 21
- `it` → `at`: 20
- `it` → `i`: 18
- `he` → `e`: 18
- `they` → `the`: 17
- `a` → `at`: 17
- `i` → `a`: 16
- `he` → `the`: 15
- `to` → `te`: 15
- `with` → `wit`: 15
- `that` → `at`: 15

Top deleted words:

- `a`: 131
- `the`: 130
- `i`: 75
- `to`: 72
- `he`: 50
- `and`: 49
- `it`: 41
- `in`: 35
- `had`: 32
- `we`: 28
- `but`: 28
- `you`: 25
- `of`: 24
- `his`: 22
- `is`: 20
- `this`: 20
- `no`: 19
- `that`: 19
- `what`: 18
- `there`: 17
- `then`: 17
- `do`: 16
- `was`: 16
- `they`: 16
- `as`: 15
- `all`: 15
- `she`: 14
- `have`: 14
- `are`: 13
- `be`: 13

Top inserted words:

- `a`: 58
- `in`: 56
- `the`: 49
- `an`: 31
- `i`: 24
- `and`: 22
- `o`: 18
- `e`: 17
- `at`: 16
- `to`: 14
- `te`: 12
- `is`: 11
- `he`: 11
- `of`: 10
- `tha`: 9
- `ther`: 9
- `his`: 8
- `as`: 8
- `de`: 8
- `it`: 8
- `ind`: 6
- `be`: 6
- `but`: 6
- `on`: 6
- `un`: 6
- `ad`: 5
- `this`: 5
- `s`: 5
- `ha`: 5
- `has`: 5

Duration-bucket WER:

- 5-10s: 0.6886
- <5s: 0.8018
- 10-15s: 0.6784

Highest speaker-level WER:

- 777: 0.7999
- 5694: 0.7976
- 422: 0.7882
- 1272: 0.7771
- 6313: 0.7722
- 2078: 0.7568
- 5338: 0.7515
- 652: 0.7413
- 3000: 0.7402
- 1988: 0.7395
- 2412: 0.7389
- 251: 0.7388
- 6241: 0.7317
- 5536: 0.7312
- 2086: 0.7302
- 2803: 0.7269
- 3853: 0.7160
- 8842: 0.7154
- 1673: 0.7080
- 1462: 0.7067
- 1919: 0.7043
- 3752: 0.7038
- 7976: 0.6967
- 2277: 0.6950
- 6345: 0.6935
- 84: 0.6889
- 174: 0.6887
- 7850: 0.6853
- 6295: 0.6821
- 3536: 0.6776

Representative high-error utterances:

- `2078-142845-0009` ref: illustration italian millet / hyp: ind i stasion at olo on on at
- `1462-170138-0003` ref: alexander exclaimed mildly / hyp: i i ha o excleindo marebly
- `2078-142845-0003` ref: seventeen eighteen / hyp: siven ty naigt ten
- `2078-142845-0041` ref: illustration buns / hyp: in was tresae bhatens
- `5694-64038-0000` ref: advance into tennessee / hyp: a vante inte con a say

## wav2vec2_discrete_k50_bilstm_ctc

Top substitutions:

- `and` → `an`: 108
- `in` → `and`: 86
- `and` → `in`: 79
- `all` → `al`: 63
- `the` → `te`: 60
- `in` → `an`: 55
- `a` → `the`: 53
- `the` → `a`: 50
- `his` → `is`: 41
- `in` → `ind`: 36
- `the` → `tha`: 33
- `is` → `as`: 33
- `is` → `his`: 31
- `the` → `he`: 30
- `to` → `the`: 28
- `that` → `at`: 26
- `they` → `the`: 25
- `as` → `is`: 24
- `will` → `wil`: 23
- `of` → `o`: 23
- `to` → `o`: 23
- `a` → `i`: 22
- `the` → `to`: 22
- `of` → `a`: 21
- `too` → `to`: 20
- `into` → `to`: 20
- `of` → `af`: 20
- `and` → `ind`: 19
- `an` → `and`: 19
- `i` → `a`: 19

Top deleted words:

- `a`: 134
- `the`: 132
- `to`: 62
- `i`: 62
- `in`: 58
- `it`: 50
- `and`: 49
- `is`: 38
- `you`: 37
- `but`: 30
- `he`: 28
- `of`: 26
- `that`: 24
- `they`: 24
- `then`: 23
- `we`: 22
- `there`: 21
- `so`: 20
- `her`: 20
- `not`: 19
- `for`: 18
- `no`: 17
- `some`: 17
- `had`: 16
- `at`: 16
- `this`: 15
- `all`: 15
- `be`: 15
- `what`: 15
- `by`: 15

Top inserted words:

- `a`: 91
- `the`: 70
- `in`: 45
- `an`: 41
- `and`: 31
- `to`: 29
- `i`: 27
- `he`: 25
- `te`: 25
- `at`: 22
- `e`: 22
- `on`: 16
- `be`: 15
- `tha`: 12
- `o`: 12
- `as`: 11
- `ind`: 11
- `ther`: 11
- `is`: 11
- `for`: 10
- `no`: 10
- `t`: 9
- `of`: 9
- `ar`: 8
- `y`: 8
- `his`: 8
- `it`: 8
- `al`: 8
- `thi`: 8
- `u`: 7

Duration-bucket WER:

- 10-15s: 0.6869
- <5s: 0.8057
- 5-10s: 0.7129
- >=15s: 0.6731

Highest speaker-level WER:

- 3570: 0.7947
- 2300: 0.7799
- 4507: 0.7729
- 5142: 0.7635
- 4446: 0.7608
- 8455: 0.7587
- 7729: 0.7567
- 8463: 0.7540
- 1188: 0.7539
- 7176: 0.7529
- 4992: 0.7496
- 1995: 0.7473
- 121: 0.7464
- 8555: 0.7459
- 237: 0.7403
- 3575: 0.7400
- 61: 0.7373
- 260: 0.7347
- 4970: 0.7184
- 2094: 0.7181
- 4077: 0.7130
- 8224: 0.7107
- 5683: 0.7069
- 6829: 0.7036
- 2961: 0.7033
- 2830: 0.6934
- 8230: 0.6912
- 7127: 0.6909
- 1320: 0.6871
- 5105: 0.6869

Representative high-error utterances:

- `1089-134691-0003` ref: the university / hyp: tha as a o vesintay
- `1995-1826-0014` ref: cotton she paused / hyp: wis it then sheme faest
- `260-123286-0010` ref: sunday august sixteenth / hyp: tsaidai l be six tive
- `5105-28241-0014` ref: another circumstance was most remarkable / hyp: and tho shor phe sent he hihe tiardil
- `672-122797-0011` ref: and then what happens then / hyp: yoe him forun wo hu pus wa ond

## wav2vec2_finetune_10h

Top substitutions:

- `it` → `it'`: 50
- `him` → `him'`: 40
- `them` → `them'`: 27
- `in` → `and`: 23
- `and` → `in`: 21
- `you` → `you'`: 19
- `me` → `me'`: 19
- `us` → `us'`: 18
- `now` → `now'`: 17
- `her` → `her'`: 17
- `home` → `home'`: 13
- `a` → `the`: 13
- `and` → `an`: 12
- `out` → `out'`: 11
- `one` → `one'`: 11
- `not` → `not'`: 11
- `all` → `all'`: 10
- `that` → `that'`: 10
- `in` → `ind`: 10
- `door` → `door'`: 9
- `life` → `life'`: 9
- `again` → `again'`: 9
- `add` → `ad`: 9
- `time` → `time'`: 9
- `away` → `away'`: 9
- `in` → `an`: 8
- `head` → `head'`: 8
- `death` → `death'`: 8
- `thing` → `thing'`: 8
- `o` → `oh`: 8

Top deleted words:

- `a`: 26
- `to`: 14
- `and`: 9
- `in`: 7
- `it`: 5
- `i`: 5
- `of`: 4
- `any`: 4
- `some`: 4
- `don`: 4
- `an`: 4
- `the`: 3
- `new`: 3
- `had`: 3
- `mary`: 3
- `em`: 3
- `were`: 2
- `under`: 2
- `us`: 2
- `there`: 2
- `red`: 2
- `every`: 2
- `gas`: 2
- `for`: 2
- `his`: 2
- `over`: 2
- `rod`: 2
- `on`: 2
- `was`: 2
- `is`: 2

Top inserted words:

- `a`: 29
- `any`: 9
- `the`: 7
- `i`: 6
- `in`: 4
- `he`: 4
- `there`: 4
- `fire`: 3
- `force`: 3
- `an`: 3
- `with`: 3
- `to`: 3
- `may`: 3
- `sweet`: 3
- `ever`: 3
- `up`: 2
- `metra`: 2
- `sause`: 2
- `which`: 2
- `down`: 2
- `foot`: 2
- `where`: 2
- `who`: 2
- `pig`: 2
- `man`: 2
- `t`: 2
- `do`: 2
- `tea`: 2
- `every`: 2
- `it`: 2

Duration-bucket WER:

- 5-10s: 0.1450
- <5s: 0.2103
- 10-15s: 0.1335

Highest speaker-level WER:

- 652: 0.2358
- 777: 0.2095
- 1272: 0.1940
- 6313: 0.1934
- 7850: 0.1862
- 3752: 0.1856
- 251: 0.1821
- 1919: 0.1807
- 2078: 0.1792
- 8842: 0.1773
- 1673: 0.1715
- 5694: 0.1668
- 2035: 0.1632
- 1988: 0.1632
- 5895: 0.1612
- 2803: 0.1603
- 2902: 0.1571
- 8297: 0.1563
- 3576: 0.1553
- 174: 0.1550
- 6295: 0.1538
- 5338: 0.1536
- 84: 0.1536
- 6241: 0.1533
- 3081: 0.1527
- 1462: 0.1513
- 2086: 0.1497
- 3000: 0.1450
- 2428: 0.1435
- 2277: 0.1408

Representative high-error utterances:

- `2078-142845-0000` ref: kirkleatham yeast / hyp: cerkley them aste'
- `2428-83705-0036` ref: someone sniggered / hyp: some one sniggared'
- `8842-304647-0001` ref: quinci impara a stupirti / hyp: queienshi am paras to berti'
- `1272-135031-0016` ref: kaliko hesitated / hyp: caligo hesitated'
- `174-50561-0014` ref: the ladies / hyp: fo ladies'

## wav2vec2_finetune_1h

Top substitutions:

- `it` → `'`: 52
- `him` → `'`: 40
- `them` → `'`: 28
- `you` → `'`: 19
- `me` → `'`: 19
- `us` → `'`: 18
- `now` → `'`: 17
- `her` → `'`: 17
- `home` → `'`: 13
- `out` → `'`: 11
- `one` → `'`: 11
- `not` → `'`: 11
- `all` → `'`: 10
- `that` → `'`: 10
- `door` → `'`: 9
- `life` → `'`: 9
- `again` → `'`: 9
- `time` → `'`: 9
- `away` → `'`: 9
- `head` → `'`: 8
- `death` → `'`: 8
- `thing` → `'`: 8
- `house` → `'`: 8
- `say` → `'`: 8
- `day` → `'`: 8
- `man` → `'`: 8
- `side` → `'`: 8
- `so` → `'`: 7
- `other` → `'`: 7
- `well` → `'`: 7

Top deleted words:

- `the`: 2741
- `and`: 1447
- `of`: 1318
- `to`: 1119
- `a`: 980
- `in`: 741
- `was`: 610
- `he`: 598
- `i`: 560
- `that`: 510
- `his`: 484
- `it`: 475
- `had`: 368
- `with`: 346
- `for`: 334
- `as`: 325
- `is`: 317
- `you`: 300
- `but`: 287
- `not`: 283
- `at`: 273
- `on`: 259
- `her`: 257
- `she`: 256
- `they`: 225
- `be`: 216
- `this`: 201
- `my`: 201
- `were`: 192
- `all`: 188

Top inserted words:


Duration-bucket WER:

- 5-10s: 1.0000
- <5s: 1.0000
- 10-15s: 1.0000

Highest speaker-level WER:

- 1272: 1.0000
- 1462: 1.0000
- 1673: 1.0000
- 174: 1.0000
- 1919: 1.0000
- 1988: 1.0000
- 1993: 1.0000
- 2035: 1.0000
- 2078: 1.0000
- 2086: 1.0000
- 2277: 1.0000
- 2412: 1.0000
- 2428: 1.0000
- 251: 1.0000
- 2803: 1.0000
- 2902: 1.0000
- 3000: 1.0000
- 3081: 1.0000
- 3170: 1.0000
- 3536: 1.0000
- 3576: 1.0000
- 3752: 1.0000
- 3853: 1.0000
- 422: 1.0000
- 5338: 1.0000
- 5536: 1.0000
- 5694: 1.0000
- 5895: 1.0000
- 6241: 1.0000
- 6295: 1.0000

Representative high-error utterances:

- `1272-128104-0000` ref: mister quilter is the apostle of the middle classes and we are glad to welcome his gospel / hyp: '
- `1272-128104-0001` ref: nor is mister quilter's manner less interesting than his matter / hyp: '
- `1272-128104-0002` ref: he tells us that at this festive season of the year with christmas and roast beef looming before us similes drawn from eating and its results occur most readily to the mind / hyp: '
- `1272-128104-0003` ref: he has grave doubts whether sir frederick leighton's work is really greek after all and can discover in it but little of rocky ithaca / hyp: '
- `1272-128104-0005` ref: it is obviously unnecessary for us to point out how luminous these criticisms are how delicate in expression / hyp: '

## wav2vec2_finetune_1h_repaired

Top substitutions:

- `all` → `al`: 84
- `been` → `ben`: 68
- `were` → `wer`: 57
- `there` → `ther`: 56
- `made` → `mad`: 55
- `are` → `ar`: 53
- `good` → `god`: 53
- `more` → `mor`: 50
- `see` → `se`: 48
- `in` → `and`: 46
- `and` → `an`: 44
- `little` → `litl`: 37
- `three` → `thre`: 36
- `said` → `sad`: 35
- `know` → `now`: 34
- `who` → `ho`: 33
- `these` → `thes`: 32
- `too` → `to`: 32
- `two` → `to`: 31
- `off` → `of`: 31
- `look` → `lok`: 30
- `like` → `lik`: 30
- `two` → `tow`: 29
- `half` → `haf`: 26
- `door` → `dor`: 26
- `before` → `befor`: 25
- `their` → `ther`: 25
- `here` → `her`: 25
- `little` → `litle`: 24
- `will` → `wil`: 24

Top deleted words:

- `a`: 37
- `the`: 20
- `and`: 16
- `in`: 15
- `i`: 14
- `to`: 14
- `an`: 12
- `are`: 10
- `of`: 9
- `it`: 9
- `will`: 8
- `well`: 7
- `all`: 7
- `with`: 7
- `do`: 6
- `you`: 6
- `had`: 5
- `can't`: 5
- `how`: 5
- `each`: 4
- `mary`: 4
- `don`: 4
- `as`: 4
- `said`: 3
- `some`: 3
- `young`: 3
- `us`: 3
- `serve`: 3
- `here`: 3
- `can`: 3

Top inserted words:

- `a`: 34
- `un`: 19
- `the`: 16
- `any`: 14
- `ther`: 14
- `in`: 9
- `and`: 7
- `over`: 7
- `for`: 7
- `every`: 7
- `some`: 7
- `my`: 7
- `to`: 6
- `no`: 5
- `be`: 5
- `gwin`: 5
- `pr`: 4
- `al`: 4
- `at`: 4
- `dis`: 4
- `an`: 4
- `mis`: 4
- `ben`: 3
- `i`: 3
- `with`: 3
- `man`: 3
- `grand`: 3
- `can`: 3
- `per`: 3
- `may`: 3

Duration-bucket WER:

- 5-10s: 0.3919
- <5s: 0.4323
- 10-15s: 0.3932

Highest speaker-level WER:

- 1272: 0.4637
- 6313: 0.4543
- 1673: 0.4519
- 777: 0.4484
- 652: 0.4424
- 1988: 0.4421
- 2078: 0.4368
- 6241: 0.4351
- 5338: 0.4295
- 3752: 0.4281
- 3000: 0.4199
- 8842: 0.4196
- 2412: 0.4174
- 5694: 0.4166
- 5895: 0.4154
- 251: 0.4137
- 7850: 0.4101
- 2086: 0.4071
- 2803: 0.4041
- 422: 0.4020
- 1919: 0.3994
- 2902: 0.3962
- 1462: 0.3939
- 3853: 0.3931
- 84: 0.3895
- 6295: 0.3872
- 6319: 0.3866
- 8297: 0.3823
- 3576: 0.3820
- 5536: 0.3813

Representative high-error utterances:

- `1919-142785-0048` ref: french forcemeat / hyp: frinch fourst meat
- `2078-142845-0000` ref: kirkleatham yeast / hyp: ceurkly them youstd
- `2428-83705-0036` ref: someone sniggered / hyp: some ons nigured
- `5895-34615-0011` ref: an everlasting laugh / hyp: and ever lasting laf
- `2428-83705-0008` ref: i gasped positively gasped / hyp: a gesetd positibly gesped d

## wav2vec2_finetune_1h_time_mask

Top substitutions:

- `all` → `al`: 81
- `been` → `ben`: 79
- `were` → `wer`: 78
- `there` → `ther`: 73
- `are` → `ar`: 61
- `see` → `se`: 58
- `more` → `mor`: 54
- `made` → `mad`: 54
- `said` → `sad`: 53
- `good` → `god`: 50
- `and` → `an`: 43
- `little` → `litl`: 39
- `their` → `ther`: 38
- `three` → `thre`: 37
- `little` → `litle`: 36
- `who` → `ho`: 35
- `too` → `to`: 35
- `in` → `and`: 35
- `one` → `on`: 34
- `off` → `of`: 34
- `like` → `lik`: 32
- `door` → `dor`: 31
- `two` → `tow`: 30
- `before` → `befor`: 29
- `here` → `her`: 29
- `look` → `lok`: 28
- `know` → `no`: 27
- `half` → `haf`: 27
- `don't` → `dont`: 27
- `with` → `wit`: 26

Top deleted words:

- `a`: 31
- `in`: 15
- `an`: 13
- `and`: 12
- `i`: 11
- `the`: 11
- `are`: 10
- `all`: 8
- `to`: 8
- `each`: 7
- `do`: 6
- `some`: 5
- `were`: 5
- `you`: 5
- `of`: 5
- `it`: 4
- `will`: 4
- `well`: 4
- `out`: 4
- `don`: 4
- `how`: 4
- `he`: 3
- `serve`: 3
- `can't`: 3
- `had`: 3
- `mary`: 3
- `we`: 3
- `more`: 3
- `one`: 3
- `can`: 3

Top inserted words:

- `un`: 25
- `a`: 25
- `any`: 17
- `the`: 14
- `ther`: 12
- `al`: 12
- `my`: 10
- `over`: 9
- `in`: 8
- `with`: 7
- `some`: 7
- `every`: 6
- `be`: 6
- `and`: 5
- `out`: 5
- `an`: 4
- `for`: 4
- `t`: 4
- `mor`: 4
- `can`: 4
- `to`: 4
- `what`: 4
- `at`: 4
- `swet`: 4
- `mis`: 4
- `ad`: 4
- `mag`: 4
- `ben`: 3
- `thats`: 3
- `i`: 3

Duration-bucket WER:

- 5-10s: 0.3921
- <5s: 0.4345
- 10-15s: 0.3974

Highest speaker-level WER:

- 1272: 0.4816
- 777: 0.4498
- 1988: 0.4494
- 652: 0.4456
- 6313: 0.4431
- 6241: 0.4400
- 2078: 0.4384
- 1673: 0.4380
- 5338: 0.4276
- 3000: 0.4262
- 5895: 0.4243
- 251: 0.4233
- 3752: 0.4202
- 8842: 0.4196
- 422: 0.4157
- 1919: 0.4148
- 5694: 0.4119
- 2412: 0.4098
- 2086: 0.4071
- 2803: 0.4007
- 1462: 0.3986
- 3853: 0.3948
- 8297: 0.3926
- 7850: 0.3921
- 6295: 0.3897
- 3576: 0.3896
- 84: 0.3895
- 5536: 0.3863
- 2902: 0.3839
- 6319: 0.3812

Representative high-error utterances:

- `2078-142845-0000` ref: kirkleatham yeast / hyp: cercly them yastd
- `174-50561-0012` ref: the wandering singer / hyp: the wat t ring stinger
- `251-137823-0024` ref: tom nodded unhappily / hyp: tome noded un haply
- `3752-4944-0044` ref: that's macklewain's business / hyp: thats macl wains busnes
- `5694-64038-0000` ref: advance into tennessee / hyp: ad vance into tona say

## wav2vec2_finetune_1h_time_mask

Top substitutions:

- `all` → `al`: 88
- `been` → `ben`: 78
- `there` → `ther`: 75
- `more` → `mor`: 75
- `are` → `ar`: 71
- `their` → `ther`: 63
- `said` → `sad`: 61
- `were` → `wer`: 61
- `good` → `god`: 56
- `see` → `se`: 53
- `little` → `litl`: 49
- `made` → `mad`: 48
- `too` → `to`: 46
- `little` → `litle`: 42
- `before` → `befor`: 42
- `and` → `an`: 42
- `who` → `ho`: 41
- `will` → `wil`: 39
- `in` → `and`: 38
- `know` → `no`: 35
- `have` → `hav`: 35
- `like` → `lik`: 34
- `one` → `on`: 34
- `know` → `now`: 33
- `i` → `iy`: 31
- `with` → `wit`: 30
- `oh` → `o`: 29
- `you` → `yo`: 29
- `room` → `rom`: 29
- `heart` → `hart`: 28

Top deleted words:

- `a`: 35
- `i`: 20
- `the`: 16
- `in`: 14
- `to`: 14
- `more`: 12
- `all`: 11
- `an`: 8
- `you`: 7
- `it`: 6
- `her`: 6
- `new`: 6
- `each`: 6
- `do`: 6
- `of`: 6
- `will`: 5
- `be`: 5
- `well`: 5
- `and`: 5
- `or`: 5
- `latter`: 5
- `fir`: 5
- `with`: 4
- `tell`: 4
- `he`: 4
- `our`: 4
- `know`: 3
- `entered`: 3
- `one`: 3
- `this`: 3

Top inserted words:

- `un`: 38
- `a`: 32
- `any`: 21
- `some`: 20
- `in`: 16
- `ther`: 14
- `al`: 13
- `the`: 12
- `every`: 11
- `how`: 9
- `with`: 9
- `up`: 8
- `over`: 8
- `be`: 6
- `ho`: 6
- `her`: 6
- `mis`: 6
- `out`: 6
- `house`: 5
- `my`: 5
- `man`: 5
- `ben`: 5
- `to`: 5
- `he`: 5
- `wher`: 4
- `our`: 4
- `what`: 4
- `o`: 4
- `for`: 4
- `of`: 4

Duration-bucket WER:

- 10-15s: 0.3940
- <5s: 0.4280
- 5-10s: 0.4001
- >=15s: 0.4147

Highest speaker-level WER:

- 7729: 0.4752
- 4077: 0.4630
- 3570: 0.4591
- 8555: 0.4502
- 4992: 0.4450
- 260: 0.4437
- 2300: 0.4403
- 4446: 0.4373
- 8463: 0.4371
- 1995: 0.4343
- 5683: 0.4332
- 6829: 0.4321
- 1188: 0.4306
- 7176: 0.4246
- 2094: 0.4154
- 3575: 0.4140
- 2961: 0.4109
- 8455: 0.4084
- 237: 0.4058
- 5142: 0.4012
- 8224: 0.4008
- 61: 0.4004
- 4970: 0.4000
- 7127: 0.3984
- 908: 0.3952
- 4507: 0.3948
- 1089: 0.3945
- 2830: 0.3932
- 5105: 0.3899
- 1284: 0.3845

Representative high-error utterances:

- `8555-292519-0002` ref: venice / hyp: e nes
- `1089-134691-0024` ref: stephanos dedalos / hyp: stafenos dead lose
- `61-70968-0038` ref: robin fitzooth / hyp: ropen fits uth
- `260-123286-0010` ref: sunday august sixteenth / hyp: sonday aogs sikt tenth
- `8463-294825-0011` ref: he's swiftly punished / hyp: he is sweiftly ponished

## wav2vec2_finetune_3h

Top substitutions:

- `in` → `and`: 25
- `and` → `in`: 19
- `in` → `ind`: 17
- `too` → `to`: 15
- `hour` → `our`: 15
- `until` → `untill`: 12
- `and` → `ind`: 12
- `and` → `an`: 11
- `food` → `fod`: 10
- `the` → `a`: 9
- `hours` → `ours`: 9
- `bread` → `bred`: 9
- `a` → `the`: 9
- `poor` → `por`: 8
- `o` → `oh`: 8
- `the` → `te`: 8
- `randal` → `randle`: 8
- `will` → `wil`: 7
- `mary` → `marry`: 7
- `burgess` → `burges`: 7
- `the` → `he`: 6
- `bartley` → `bartly`: 6
- `into` → `to`: 6
- `add` → `ad`: 6
- `in` → `an`: 6
- `phoebe` → `feby`: 6
- `mood` → `mod`: 6
- `george` → `jorge`: 6
- `guard` → `gard`: 5
- `through` → `though`: 5

Top deleted words:

- `a`: 25
- `and`: 14
- `to`: 13
- `the`: 10
- `in`: 10
- `some`: 8
- `all`: 6
- `are`: 5
- `don`: 5
- `of`: 4
- `it`: 4
- `an`: 4
- `i`: 4
- `mary`: 4
- `no`: 3
- `there`: 3
- `with`: 3
- `any`: 3
- `new`: 3
- `up`: 3
- `one`: 3
- `on`: 3
- `mac`: 2
- `under`: 2
- `been`: 2
- `old`: 2
- `white`: 2
- `you're`: 2
- `i'll`: 2
- `dining`: 2

Top inserted words:

- `a`: 24
- `any`: 14
- `in`: 9
- `and`: 8
- `the`: 8
- `he`: 6
- `sweet`: 5
- `some`: 4
- `over`: 4
- `every`: 4
- `an`: 4
- `with`: 4
- `be`: 4
- `where`: 4
- `there`: 4
- `for`: 4
- `down`: 3
- `t`: 3
- `out`: 3
- `no`: 3
- `table`: 3
- `af`: 2
- `main`: 2
- `up`: 2
- `held`: 2
- `f`: 2
- `grand`: 2
- `ben`: 2
- `may`: 2
- `who`: 2

Duration-bucket WER:

- 5-10s: 0.1621
- <5s: 0.1761
- 10-15s: 0.1674

Highest speaker-level WER:

- 652: 0.2500
- 777: 0.2288
- 1272: 0.2199
- 1673: 0.2178
- 6313: 0.2115
- 1988: 0.1969
- 1919: 0.1930
- 251: 0.1897
- 8842: 0.1869
- 2078: 0.1856
- 3752: 0.1817
- 5694: 0.1787
- 3576: 0.1766
- 2086: 0.1764
- 2902: 0.1763
- 2803: 0.1693
- 5536: 0.1679
- 6241: 0.1675
- 5338: 0.1673
- 5895: 0.1647
- 84: 0.1646
- 1462: 0.1607
- 3000: 0.1602
- 7850: 0.1592
- 6295: 0.1564
- 2035: 0.1543
- 8297: 0.1535
- 174: 0.1512
- 422: 0.1510
- 2412: 0.1494

Representative high-error utterances:

- `2078-142845-0000` ref: kirkleatham yeast / hyp: curkley them yast
- `1919-142785-0046` ref: illustration basil / hyp: ilustration bazsel
- `1919-142785-0048` ref: french forcemeat / hyp: french forst meet
- `2078-142845-0022` ref: excellent rolls / hyp: ecellent grols
- `251-136532-0022` ref: lectures / hyp: lecturs

## wav2vec2_finetune_3h

Top substitutions:

- `and` → `in`: 30
- `in` → `and`: 29
- `too` → `to`: 25
- `christ` → `crist`: 21
- `an` → `and`: 15
- `the` → `a`: 13
- `thee` → `the`: 12
- `hours` → `ours`: 11
- `a` → `the`: 11
- `and` → `ind`: 10
- `and` → `an`: 10
- `in` → `ind`: 10
- `until` → `untill`: 10
- `paul` → `pal`: 10
- `tree` → `trey`: 10
- `hour` → `our`: 9
- `wrong` → `rong`: 9
- `oh` → `o`: 9
- `in` → `an`: 8
- `greatly` → `greately`: 8
- `eye` → `ey`: 8
- `will` → `wil`: 8
- `united` → `unighted`: 8
- `consumption` → `conscemption`: 8
- `knife` → `nife`: 8
- `their` → `thir`: 7
- `character` → `caracter`: 7
- `carefully` → `carfully`: 7
- `cotton` → `coten`: 7
- `carriage` → `carrage`: 7

Top deleted words:

- `a`: 26
- `in`: 14
- `to`: 13
- `and`: 12
- `the`: 10
- `it`: 6
- `on`: 6
- `up`: 5
- `i`: 5
- `of`: 5
- `with`: 5
- `new`: 5
- `you`: 4
- `all`: 4
- `some`: 4
- `have`: 4
- `latter`: 4
- `school`: 3
- `more`: 3
- `rose`: 3
- `other`: 3
- `were`: 3
- `is`: 3
- `this`: 3
- `or`: 3
- `ben`: 3
- `an`: 3
- `mademoiselle`: 3
- `silk`: 2
- `over`: 2

Top inserted words:

- `a`: 22
- `in`: 13
- `any`: 13
- `the`: 11
- `every`: 10
- `there`: 7
- `un`: 7
- `to`: 7
- `for`: 7
- `where`: 6
- `some`: 6
- `i`: 6
- `all`: 6
- `up`: 5
- `with`: 5
- `is`: 5
- `t`: 4
- `be`: 4
- `what`: 4
- `o`: 4
- `over`: 4
- `and`: 4
- `re`: 4
- `he`: 4
- `fire`: 3
- `on`: 3
- `forth`: 3
- `it`: 3
- `an`: 3
- `as`: 3

Duration-bucket WER:

- 10-15s: 0.1653
- <5s: 0.1841
- 5-10s: 0.1679
- >=15s: 0.1715

Highest speaker-level WER:

- 7176: 0.2246
- 3570: 0.2215
- 8555: 0.2184
- 7729: 0.2172
- 2961: 0.2163
- 4992: 0.2103
- 2300: 0.2087
- 8463: 0.2032
- 4077: 0.2029
- 1995: 0.2019
- 2830: 0.1970
- 4970: 0.1903
- 61: 0.1884
- 260: 0.1870
- 6829: 0.1857
- 4507: 0.1854
- 908: 0.1775
- 1089: 0.1684
- 1188: 0.1667
- 2094: 0.1655
- 5105: 0.1635
- 8455: 0.1628
- 5142: 0.1611
- 121: 0.1610
- 5683: 0.1570
- 4446: 0.1562
- 3575: 0.1533
- 237: 0.1525
- 1284: 0.1508
- 7127: 0.1494

Representative high-error utterances:

- `1089-134691-0024` ref: stephanos dedalos / hyp: staffeno stdead los
- `3575-170457-0016` ref: farewell madam / hyp: fair well madame
- `61-70968-0038` ref: robin fitzooth / hyp: robpen fits of
- `237-134500-0001` ref: marie sighed / hyp: marrie siged
- `237-134500-0025` ref: oh emil / hyp: o amial

## wav2vec2_frozen_10h_fair_30ep

Top substitutions:

- `the` → `n`: 43
- `the` → `s`: 28
- `the` → `t`: 28
- `and` → `n`: 24
- `of` → `n`: 18
- `to` → `n`: 18
- `to` → `s`: 16
- `of` → `s`: 15
- `the` → `m`: 13
- `of` → `d`: 13
- `her` → `n`: 12
- `in` → `n`: 12
- `of` → `t`: 11
- `was` → `n`: 11
- `and` → `s`: 11
- `a` → `n`: 11
- `and` → `d`: 10
- `the` → `r`: 10
- `to` → `t`: 9
- `it` → `n`: 9
- `that` → `n`: 9
- `the` → `ns`: 9
- `of` → `r`: 9
- `the` → `nn`: 8
- `that` → `s`: 7
- `the` → `d`: 7
- `a` → `s`: 7
- `his` → `s`: 7
- `for` → `n`: 7
- `as` → `n`: 7

Top deleted words:

- `the`: 2224
- `and`: 1195
- `of`: 985
- `to`: 874
- `a`: 825
- `in`: 601
- `he`: 550
- `was`: 529
- `i`: 512
- `that`: 432
- `it`: 414
- `his`: 383
- `had`: 316
- `you`: 281
- `with`: 275
- `is`: 273
- `but`: 263
- `for`: 262
- `as`: 260
- `not`: 239
- `at`: 233
- `she`: 227
- `on`: 209
- `her`: 193
- `they`: 188
- `be`: 175
- `this`: 174
- `my`: 168
- `were`: 159
- `we`: 152

Top inserted words:


Duration-bucket WER:

- 5-10s: 0.9996
- <5s: 0.9997
- 10-15s: 0.9985

Highest speaker-level WER:

- 174: 1.0000
- 1919: 1.0000
- 1993: 1.0000
- 2035: 1.0000
- 2078: 1.0000
- 2086: 1.0000
- 2428: 1.0000
- 251: 1.0000
- 3752: 1.0000
- 3853: 1.0000
- 5536: 1.0000
- 5694: 1.0000
- 5895: 1.0000
- 6241: 1.0000
- 6295: 1.0000
- 6345: 1.0000
- 777: 1.0000
- 7850: 1.0000
- 8297: 1.0000
- 2277: 0.9994
- 3536: 0.9993
- 7976: 0.9993
- 2412: 0.9992
- 1462: 0.9992
- 84: 0.9992
- 652: 0.9992
- 3081: 0.9991
- 8842: 0.9991
- 5338: 0.9990
- 1272: 0.9990

Representative high-error utterances:

- `1272-128104-0000` ref: mister quilter is the apostle of the middle classes and we are glad to welcome his gospel / hyp: s t t  mtn dws
- `1272-128104-0001` ref: nor is mister quilter's manner less interesting than his matter / hyp: ssmnsntnst
- `1272-128104-0003` ref: he has grave doubts whether sir frederick leighton's work is really greek after all and can discover in it but little of rocky ithaca / hyp: s  sswtsfdcnsstrfrlnndstrrn tdls
- `1272-128104-0005` ref: it is obviously unnecessary for us to point out how luminous these criticisms are how delicate in expression / hyp: snssfst tt mnssstsondlcd nn
- `1272-128104-0006` ref: on the general principles of art mister quilter writes with equal lucidity / hyp: ndnrssrnmstorrslt

## wav2vec2_frozen_10h_v2

Top substitutions:


Top deleted words:

- `the`: 3154
- `and`: 1726
- `of`: 1571
- `to`: 1284
- `a`: 1108
- `in`: 861
- `was`: 662
- `he`: 659
- `i`: 608
- `that`: 595
- `it`: 589
- `his`: 557
- `with`: 420
- `had`: 418
- `as`: 393
- `for`: 373
- `is`: 350
- `you`: 339
- `not`: 324
- `but`: 318
- `her`: 310
- `on`: 303
- `at`: 301
- `she`: 275
- `they`: 248
- `be`: 245
- `my`: 229
- `all`: 222
- `were`: 221
- `this`: 219

Top inserted words:


Duration-bucket WER:

- 5-10s: 1.0000
- <5s: 1.0000
- 10-15s: 1.0000
- >=15s: 1.0000

Highest speaker-level WER:

- 1272: 1.0000
- 1462: 1.0000
- 1673: 1.0000
- 174: 1.0000
- 1919: 1.0000
- 1988: 1.0000
- 1993: 1.0000
- 2035: 1.0000
- 2078: 1.0000
- 2086: 1.0000
- 2277: 1.0000
- 2412: 1.0000
- 2428: 1.0000
- 251: 1.0000
- 2803: 1.0000
- 2902: 1.0000
- 3000: 1.0000
- 3081: 1.0000
- 3170: 1.0000
- 3536: 1.0000
- 3576: 1.0000
- 3752: 1.0000
- 3853: 1.0000
- 422: 1.0000
- 5338: 1.0000
- 5536: 1.0000
- 5694: 1.0000
- 5895: 1.0000
- 6241: 1.0000
- 6295: 1.0000

Representative high-error utterances:

- `1272-128104-0000` ref: mister quilter is the apostle of the middle classes and we are glad to welcome his gospel / hyp: 
- `1272-128104-0001` ref: nor is mister quilter's manner less interesting than his matter / hyp: 
- `1272-128104-0002` ref: he tells us that at this festive season of the year with christmas and roast beef looming before us similes drawn from eating and its results occur most readily to the mind / hyp: 
- `1272-128104-0003` ref: he has grave doubts whether sir frederick leighton's work is really greek after all and can discover in it but little of rocky ithaca / hyp: 
- `1272-128104-0005` ref: it is obviously unnecessary for us to point out how luminous these criticisms are how delicate in expression / hyp: 

## wav2vec2_frozen_bilstm_10h

Top substitutions:

- `it` → `'`: 52
- `him` → `'`: 40
- `them` → `'`: 28
- `you` → `'`: 19
- `me` → `'`: 19
- `us` → `'`: 18
- `now` → `'`: 17
- `her` → `'`: 17
- `home` → `'`: 13
- `out` → `'`: 11
- `one` → `'`: 11
- `not` → `'`: 11
- `all` → `'`: 10
- `that` → `'`: 10
- `door` → `'`: 9
- `life` → `'`: 9
- `again` → `'`: 9
- `time` → `'`: 9
- `away` → `'`: 9
- `so` → `'`: 8
- `head` → `'`: 8
- `death` → `'`: 8
- `thing` → `'`: 8
- `house` → `'`: 8
- `say` → `'`: 8
- `day` → `'`: 8
- `man` → `'`: 8
- `side` → `'`: 8
- `other` → `'`: 7
- `well` → `'`: 7

Top deleted words:

- `the`: 2741
- `and`: 1447
- `of`: 1318
- `to`: 1119
- `a`: 980
- `in`: 741
- `was`: 610
- `he`: 598
- `i`: 560
- `that`: 510
- `his`: 484
- `it`: 475
- `had`: 368
- `with`: 346
- `for`: 334
- `as`: 325
- `is`: 317
- `you`: 300
- `but`: 287
- `not`: 283
- `at`: 273
- `on`: 259
- `her`: 257
- `she`: 256
- `they`: 225
- `be`: 216
- `this`: 201
- `my`: 201
- `were`: 192
- `all`: 188

Top inserted words:


Duration-bucket WER:

- 5-10s: 1.0000
- <5s: 1.0000
- 10-15s: 1.0000

Highest speaker-level WER:

- 1272: 1.0000
- 1462: 1.0000
- 1673: 1.0000
- 174: 1.0000
- 1919: 1.0000
- 1988: 1.0000
- 1993: 1.0000
- 2035: 1.0000
- 2078: 1.0000
- 2086: 1.0000
- 2277: 1.0000
- 2412: 1.0000
- 2428: 1.0000
- 251: 1.0000
- 2803: 1.0000
- 2902: 1.0000
- 3000: 1.0000
- 3081: 1.0000
- 3170: 1.0000
- 3536: 1.0000
- 3576: 1.0000
- 3752: 1.0000
- 3853: 1.0000
- 422: 1.0000
- 5338: 1.0000
- 5536: 1.0000
- 5694: 1.0000
- 5895: 1.0000
- 6241: 1.0000
- 6295: 1.0000

Representative high-error utterances:

- `1272-128104-0000` ref: mister quilter is the apostle of the middle classes and we are glad to welcome his gospel / hyp: '
- `1272-128104-0001` ref: nor is mister quilter's manner less interesting than his matter / hyp: '
- `1272-128104-0002` ref: he tells us that at this festive season of the year with christmas and roast beef looming before us similes drawn from eating and its results occur most readily to the mind / hyp: '
- `1272-128104-0003` ref: he has grave doubts whether sir frederick leighton's work is really greek after all and can discover in it but little of rocky ithaca / hyp: '
- `1272-128104-0005` ref: it is obviously unnecessary for us to point out how luminous these criticisms are how delicate in expression / hyp: '

## wav2vec2_frozen_bilstm_10h

Top substitutions:

- `it` → `'`: 36
- `him` → `'`: 35
- `me` → `'`: 35
- `you` → `'`: 23
- `them` → `'`: 19
- `said` → `'`: 15
- `time` → `'`: 15
- `her` → `'`: 13
- `life` → `'`: 13
- `man` → `'`: 12
- `day` → `'`: 12
- `tree` → `'`: 12
- `all` → `'`: 11
- `now` → `'`: 11
- `room` → `'`: 11
- `this` → `'`: 11
- `up` → `'`: 10
- `world` → `'`: 10
- `there` → `'`: 10
- `moment` → `'`: 9
- `sir` → `'`: 9
- `so` → `'`: 8
- `way` → `'`: 8
- `head` → `'`: 8
- `that` → `'`: 8
- `slang` → `'`: 8
- `on` → `'`: 7
- `face` → `'`: 7
- `light` → `'`: 7
- `fire` → `'`: 7

Top deleted words:

- `the`: 3461
- `of`: 1796
- `and`: 1787
- `to`: 1337
- `a`: 1166
- `in`: 900
- `i`: 709
- `that`: 602
- `was`: 576
- `he`: 524
- `it`: 522
- `his`: 473
- `is`: 462
- `with`: 423
- `for`: 417
- `you`: 395
- `as`: 383
- `but`: 344
- `not`: 332
- `had`: 321
- `her`: 311
- `be`: 310
- `at`: 283
- `she`: 279
- `on`: 272
- `this`: 252
- `by`: 243
- `my`: 225
- `which`: 215
- `have`: 215

Top inserted words:


Duration-bucket WER:

- 10-15s: 1.0000
- <5s: 1.0000
- 5-10s: 1.0000
- >=15s: 1.0000

Highest speaker-level WER:

- 1089: 1.0000
- 1188: 1.0000
- 121: 1.0000
- 1221: 1.0000
- 1284: 1.0000
- 1320: 1.0000
- 1580: 1.0000
- 1995: 1.0000
- 2094: 1.0000
- 2300: 1.0000
- 237: 1.0000
- 260: 1.0000
- 2830: 1.0000
- 2961: 1.0000
- 3570: 1.0000
- 3575: 1.0000
- 3729: 1.0000
- 4077: 1.0000
- 4446: 1.0000
- 4507: 1.0000
- 4970: 1.0000
- 4992: 1.0000
- 5105: 1.0000
- 5142: 1.0000
- 5639: 1.0000
- 5683: 1.0000
- 61: 1.0000
- 672: 1.0000
- 6829: 1.0000
- 6930: 1.0000

Representative high-error utterances:

- `1089-134686-0000` ref: he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick peppered flour fattened sauce / hyp: '
- `1089-134686-0001` ref: stuff it into you his belly counselled him / hyp: '
- `1089-134686-0002` ref: after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels / hyp: '
- `1089-134686-0003` ref: hello bertie any good in your mind / hyp: '
- `1089-134686-0004` ref: number ten fresh nelly is waiting on you good night husband / hyp: '

## wav2vec2_layer12_bilstm_ctc

Top substitutions:

- `and` → `an`: 79
- `in` → `an`: 50
- `in` → `and`: 48
- `and` → `in`: 45
- `his` → `is`: 40
- `two` → `to`: 32
- `in` → `ind`: 32
- `is` → `his`: 29
- `are` → `ar`: 27
- `into` → `to`: 27
- `it` → `at`: 25
- `the` → `he`: 24
- `all` → `al`: 24
- `too` → `to`: 22
- `they` → `the`: 22
- `the` → `te`: 21
- `is` → `as`: 21
- `been` → `ben`: 20
- `their` → `ther`: 20
- `an` → `and`: 19
- `door` → `dor`: 18
- `three` → `thre`: 18
- `at` → `it`: 18
- `there` → `ther`: 18
- `had` → `ad`: 17
- `he` → `e`: 17
- `know` → `no`: 16
- `when` → `wen`: 16
- `of` → `a`: 16
- `as` → `is`: 16

Top deleted words:

- `a`: 141
- `the`: 36
- `i`: 32
- `in`: 29
- `to`: 28
- `and`: 20
- `he`: 18
- `all`: 18
- `there`: 16
- `you`: 13
- `on`: 12
- `are`: 12
- `for`: 12
- `of`: 12
- `some`: 11
- `our`: 11
- `they`: 11
- `as`: 11
- `we`: 10
- `be`: 10
- `it`: 10
- `were`: 10
- `an`: 10
- `can`: 9
- `had`: 9
- `is`: 9
- `see`: 9
- `no`: 8
- `will`: 8
- `his`: 8

Top inserted words:

- `a`: 103
- `in`: 37
- `an`: 28
- `to`: 21
- `the`: 18
- `and`: 14
- `for`: 12
- `over`: 11
- `with`: 11
- `i`: 11
- `ar`: 10
- `it`: 10
- `at`: 9
- `be`: 9
- `some`: 9
- `o`: 8
- `t`: 7
- `un`: 7
- `any`: 7
- `as`: 6
- `on`: 6
- `he`: 6
- `de`: 5
- `re`: 5
- `every`: 5
- `do`: 5
- `is`: 5
- `af`: 5
- `there`: 5
- `gin`: 5

Duration-bucket WER:

- 5-10s: 0.4471
- <5s: 0.5771
- 10-15s: 0.4299

Highest speaker-level WER:

- 6313: 0.5701
- 5694: 0.5597
- 777: 0.5581
- 1272: 0.5552
- 1988: 0.5289
- 652: 0.5237
- 6241: 0.5216
- 5338: 0.5196
- 5536: 0.5021
- 422: 0.5000
- 3752: 0.4984
- 2803: 0.4898
- 2078: 0.4880
- 8842: 0.4853
- 3000: 0.4805
- 5895: 0.4774
- 1462: 0.4719
- 251: 0.4700
- 3170: 0.4693
- 1919: 0.4692
- 2412: 0.4679
- 3081: 0.4616
- 7976: 0.4550
- 2086: 0.4528
- 2277: 0.4491
- 2902: 0.4450
- 1673: 0.4450
- 3576: 0.4444
- 84: 0.4444
- 3853: 0.4433

Representative high-error utterances:

- `3081-166546-0021` ref: i emphasised complacently / hyp: iy amp the cice d com playesonetlig
- `2078-142845-0044` ref: italian rusks / hyp: at tally in rusps
- `251-136532-0022` ref: lectures / hyp: whli tures
- `3081-166546-0005` ref: george nodded / hyp: jor tu not it
- `8297-275155-0021` ref: honestly / hyp: on estly

## wav2vec2_layer12_bilstm_ctc

Top substitutions:

- `and` → `an`: 79
- `and` → `in`: 70
- `in` → `and`: 67
- `in` → `an`: 58
- `are` → `ar`: 40
- `is` → `as`: 34
- `will` → `wil`: 33
- `too` → `to`: 32
- `an` → `and`: 31
- `is` → `his`: 31
- `it` → `at`: 27
- `the` → `te`: 25
- `as` → `is`: 22
- `thee` → `the`: 22
- `there` → `ther`: 21
- `all` → `al`: 20
- `in` → `ind`: 20
- `three` → `thre`: 20
- `their` → `ther`: 19
- `two` → `to`: 18
- `know` → `now`: 18
- `a` → `as`: 18
- `too` → `two`: 18
- `still` → `stil`: 17
- `his` → `is`: 17
- `our` → `ar`: 17
- `been` → `ben`: 16
- `of` → `o`: 16
- `see` → `se`: 16
- `the` → `a`: 16

Top deleted words:

- `a`: 163
- `in`: 40
- `i`: 40
- `and`: 32
- `the`: 27
- `to`: 24
- `you`: 23
- `be`: 23
- `it`: 22
- `of`: 18
- `for`: 16
- `all`: 14
- `we`: 14
- `her`: 13
- `with`: 13
- `there`: 11
- `an`: 11
- `they`: 10
- `can`: 10
- `or`: 10
- `but`: 9
- `is`: 9
- `my`: 9
- `do`: 9
- `on`: 8
- `are`: 8
- `know`: 8
- `oh`: 8
- `am`: 8
- `will`: 8

Top inserted words:

- `a`: 106
- `in`: 44
- `an`: 27
- `and`: 23
- `the`: 19
- `e`: 19
- `i`: 16
- `all`: 14
- `as`: 14
- `to`: 14
- `o`: 13
- `on`: 12
- `be`: 12
- `some`: 12
- `at`: 12
- `he`: 12
- `is`: 10
- `un`: 10
- `over`: 10
- `so`: 9
- `de`: 8
- `any`: 8
- `or`: 8
- `t`: 7
- `ad`: 7
- `you`: 7
- `with`: 7
- `no`: 6
- `her`: 6
- `ar`: 6

Duration-bucket WER:

- 10-15s: 0.4180
- <5s: 0.5724
- 5-10s: 0.4607
- >=15s: 0.4156

Highest speaker-level WER:

- 3570: 0.5631
- 1995: 0.5485
- 4446: 0.5412
- 5142: 0.5281
- 4992: 0.5201
- 8463: 0.5137
- 260: 0.5102
- 8455: 0.5036
- 3575: 0.4993
- 8555: 0.4993
- 1188: 0.4985
- 4077: 0.4985
- 7176: 0.4976
- 2830: 0.4921
- 2961: 0.4888
- 2094: 0.4884
- 7729: 0.4874
- 4507: 0.4854
- 2300: 0.4738
- 61: 0.4652
- 237: 0.4633
- 6829: 0.4632
- 8224: 0.4536
- 1580: 0.4522
- 121: 0.4413
- 4970: 0.4352
- 5683: 0.4348
- 6930: 0.4311
- 3729: 0.4287
- 1284: 0.4222

Representative high-error utterances:

- `1089-134691-0003` ref: the university / hyp: this yun of vrsity
- `3575-170457-0016` ref: farewell madam / hyp: far lom that emn
- `8555-284447-0016` ref: fine glorious / hyp: fom erim o ras
- `8555-292519-0002` ref: venice / hyp: ti is
- `3729-6852-0010` ref: i never had any family / hyp: bhyn mo r h a mon het ambee

## wav2vec2_layer6_bilstm_ctc

Top substitutions:

- `in` → `and`: 62
- `and` → `an`: 54
- `and` → `in`: 50
- `in` → `an`: 28
- `it` → `at`: 18
- `four` → `for`: 15
- `his` → `is`: 14
- `in` → `ind`: 14
- `into` → `to`: 13
- `are` → `ar`: 13
- `there` → `ther`: 13
- `is` → `as`: 13
- `too` → `two`: 12
- `to` → `a`: 12
- `at` → `it`: 12
- `to` → `o`: 12
- `of` → `o`: 11
- `is` → `his`: 11
- `a` → `the`: 11
- `the` → `a`: 10
- `you` → `ou`: 10
- `it's` → `its`: 10
- `been` → `ben`: 10
- `and` → `ind`: 10
- `a` → `an`: 10
- `of` → `a`: 10
- `they` → `the`: 9
- `you're` → `your`: 9
- `bartley` → `bartly`: 9
- `a` → `o`: 9

Top deleted words:

- `a`: 71
- `in`: 34
- `to`: 23
- `and`: 23
- `the`: 17
- `i`: 11
- `of`: 10
- `is`: 10
- `on`: 7
- `you`: 7
- `it`: 7
- `an`: 7
- `there`: 6
- `were`: 6
- `mary`: 6
- `we`: 6
- `all`: 6
- `he`: 5
- `this`: 5
- `under`: 5
- `each`: 5
- `as`: 5
- `some`: 5
- `any`: 5
- `where`: 4
- `with`: 4
- `they`: 4
- `how`: 4
- `one`: 4
- `shall`: 4

Top inserted words:

- `a`: 55
- `in`: 21
- `the`: 18
- `i`: 16
- `to`: 13
- `over`: 12
- `be`: 10
- `some`: 9
- `an`: 9
- `and`: 9
- `with`: 9
- `for`: 7
- `he`: 7
- `s`: 6
- `at`: 6
- `any`: 6
- `e`: 6
- `of`: 5
- `grand`: 5
- `no`: 5
- `un`: 5
- `as`: 5
- `man`: 5
- `o`: 4
- `ind`: 4
- `hild`: 4
- `is`: 4
- `al`: 4
- `there`: 4
- `t`: 3

Duration-bucket WER:

- 5-10s: 0.2571
- <5s: 0.4103
- 10-15s: 0.2471

Highest speaker-level WER:

- 777: 0.3859
- 1272: 0.3761
- 6313: 0.3655
- 2803: 0.3488
- 5694: 0.3447
- 1988: 0.3424
- 6241: 0.3215
- 652: 0.3194
- 5536: 0.3151
- 2078: 0.3120
- 3853: 0.3108
- 2412: 0.3057
- 5338: 0.3043
- 1673: 0.3013
- 1919: 0.2998
- 174: 0.2988
- 5895: 0.2976
- 7850: 0.2950
- 2902: 0.2932
- 3752: 0.2923
- 2086: 0.2869
- 8842: 0.2846
- 251: 0.2831
- 2428: 0.2810
- 422: 0.2804
- 3081: 0.2744
- 6319: 0.2683
- 84: 0.2680
- 3000: 0.2673
- 1462: 0.2668

Representative high-error utterances:

- `1919-142785-0048` ref: french forcemeat / hyp: friindo for is net
- `2078-142845-0009` ref: illustration italian millet / hyp: il strtion ei ty an melet
- `8842-304647-0007` ref: most wonderful / hyp: mos o mond iuffo
- `2078-142845-0048` ref: seventeen thirty four / hyp: sv hin ten theirty foar
- `3081-166546-0021` ref: i emphasised complacently / hyp: amthe ss t complay soentli

## wav2vec2_layer6_bilstm_ctc

Top substitutions:

- `and` → `in`: 70
- `and` → `an`: 62
- `in` → `and`: 51
- `in` → `an`: 31
- `an` → `and`: 26
- `is` → `as`: 21
- `are` → `ar`: 21
- `it` → `at`: 20
- `in` → `ind`: 18
- `thee` → `the`: 18
- `is` → `his`: 16
- `too` → `two`: 16
- `and` → `ind`: 14
- `of` → `o`: 14
- `too` → `to`: 12
- `where` → `were`: 12
- `there` → `ther`: 11
- `of` → `a`: 11
- `as` → `is`: 11
- `a` → `the`: 11
- `the` → `he`: 10
- `the` → `a`: 10
- `a` → `o`: 10
- `this` → `the`: 10
- `you` → `ou`: 10
- `to` → `a`: 10
- `bartley` → `bartly`: 10
- `will` → `wil`: 9
- `character` → `caracter`: 9
- `they` → `the`: 9

Top deleted words:

- `a`: 84
- `in`: 27
- `the`: 24
- `to`: 20
- `and`: 19
- `i`: 17
- `he`: 16
- `you`: 13
- `or`: 12
- `of`: 11
- `but`: 10
- `at`: 8
- `more`: 8
- `all`: 8
- `be`: 8
- `do`: 8
- `is`: 7
- `for`: 7
- `can`: 6
- `one`: 6
- `so`: 5
- `then`: 5
- `it`: 5
- `his`: 5
- `up`: 5
- `out`: 4
- `that`: 4
- `not`: 4
- `how`: 4
- `why`: 4

Top inserted words:

- `a`: 67
- `in`: 36
- `the`: 21
- `an`: 20
- `be`: 13
- `and`: 13
- `i`: 12
- `s`: 12
- `to`: 11
- `with`: 11
- `any`: 10
- `o`: 10
- `e`: 10
- `all`: 9
- `ar`: 9
- `as`: 8
- `there`: 8
- `some`: 7
- `for`: 7
- `he`: 7
- `never`: 7
- `on`: 6
- `tha`: 6
- `at`: 6
- `over`: 6
- `what`: 5
- `ther`: 5
- `n`: 5
- `al`: 5
- `up`: 4

Duration-bucket WER:

- 10-15s: 0.2544
- <5s: 0.4316
- 5-10s: 0.2961
- >=15s: 0.2435

Highest speaker-level WER:

- 1995: 0.4006
- 7176: 0.3863
- 8555: 0.3759
- 3570: 0.3741
- 2961: 0.3659
- 7729: 0.3493
- 8463: 0.3427
- 4992: 0.3380
- 4077: 0.3341
- 4507: 0.3333
- 260: 0.3294
- 6829: 0.3261
- 2300: 0.3257
- 8455: 0.3241
- 4446: 0.3190
- 2830: 0.3182
- 3575: 0.3159
- 4970: 0.3146
- 5142: 0.3108
- 61: 0.3045
- 2094: 0.2983
- 1188: 0.2909
- 5105: 0.2901
- 6930: 0.2872
- 1284: 0.2808
- 7127: 0.2743
- 908: 0.2736
- 237: 0.2734
- 5683: 0.2657
- 121: 0.2633

Representative high-error utterances:

- `1089-134691-0003` ref: the university / hyp: they yeuwn o varsitay
- `1089-134691-0024` ref: stephanos dedalos / hyp: stiffin nos deit los
- `2094-142345-0016` ref: spinning indeed / hyp: sphai ai neain theid
- `3575-170457-0016` ref: farewell madam / hyp: fear ba med m
- `1995-1836-0011` ref: positively heroic added cresswell avoiding his sister's eyes / hyp: bos tha ge thea ghr allict ad it coures wel la oirtyg hes scisthors ais

## wav2vec2_layer9_bilstm_ctc

Top substitutions:

- `and` → `an`: 31
- `in` → `and`: 25
- `and` → `in`: 17
- `in` → `an`: 15
- `into` → `to`: 13
- `four` → `for`: 12
- `a` → `the`: 11
- `the` → `a`: 10
- `his` → `is`: 9
- `add` → `ad`: 9
- `two` → `to`: 9
- `is` → `as`: 9
- `i'm` → `im`: 8
- `and` → `ind`: 8
- `are` → `ar`: 8
- `to` → `o`: 7
- `it's` → `its`: 7
- `of` → `o`: 7
- `eggs` → `egs`: 7
- `their` → `ther`: 7
- `a` → `ha`: 7
- `bozzle` → `bosl`: 7
- `burgess` → `burges`: 7
- `too` → `to`: 6
- `you're` → `youre`: 6
- `bartley` → `bartle`: 6
- `you` → `ou`: 6
- `o` → `oh`: 6
- `flour` → `flower`: 6
- `been` → `ben`: 6

Top deleted words:

- `a`: 35
- `in`: 17
- `the`: 11
- `i`: 10
- `and`: 8
- `is`: 6
- `this`: 6
- `to`: 6
- `an`: 6
- `will`: 4
- `there`: 4
- `shall`: 4
- `do`: 3
- `it`: 3
- `he`: 3
- `some`: 3
- `who`: 3
- `are`: 3
- `these`: 3
- `how`: 3
- `mary`: 3
- `every`: 3
- `soon`: 3
- `any`: 3
- `vita`: 3
- `where`: 2
- `of`: 2
- `mac`: 2
- `oh`: 2
- `us`: 2

Top inserted words:

- `a`: 32
- `in`: 22
- `the`: 12
- `s`: 8
- `any`: 7
- `he`: 6
- `for`: 6
- `i`: 6
- `over`: 6
- `an`: 5
- `grand`: 5
- `no`: 5
- `af`: 4
- `is`: 4
- `some`: 4
- `with`: 4
- `and`: 4
- `mag`: 4
- `o`: 3
- `man`: 3
- `fire`: 3
- `al`: 3
- `at`: 3
- `every`: 3
- `ar`: 3
- `of`: 3
- `there`: 3
- `sweet`: 3
- `table`: 3
- `up`: 2

Duration-bucket WER:

- 5-10s: 0.1629
- <5s: 0.3049
- 10-15s: 0.1557

Highest speaker-level WER:

- 777: 0.2690
- 652: 0.2626
- 1272: 0.2418
- 5694: 0.2379
- 1988: 0.2371
- 251: 0.2345
- 1919: 0.2300
- 5895: 0.2250
- 6313: 0.2222
- 3752: 0.2196
- 2086: 0.2145
- 6241: 0.2087
- 7850: 0.2077
- 1673: 0.2074
- 174: 0.2037
- 2078: 0.2000
- 2902: 0.1955
- 3081: 0.1941
- 2428: 0.1901
- 8842: 0.1894
- 2803: 0.1874
- 84: 0.1857
- 5536: 0.1853
- 6295: 0.1821
- 5338: 0.1800
- 2412: 0.1781
- 1462: 0.1732
- 3853: 0.1723
- 2035: 0.1722
- 3576: 0.1720

Representative high-error utterances:

- `1919-142785-0048` ref: french forcemeat / hyp: friche for ch meat
- `2078-142845-0009` ref: illustration italian millet / hyp: illstration i tai an mel at
- `251-136532-0022` ref: lectures / hyp: lck ers
- `8842-304647-0007` ref: most wonderful / hyp: miols sto and theffal
- `1919-142785-0039` ref: illustration marjoram / hyp: illstration my rham

## wav2vec2_layer9_bilstm_ctc

Top substitutions:

- `and` → `in`: 32
- `and` → `an`: 31
- `in` → `and`: 23
- `too` → `to`: 18
- `an` → `and`: 14
- `in` → `an`: 14
- `paul` → `pall`: 14
- `christ` → `criste`: 12
- `a` → `the`: 12
- `is` → `as`: 11
- `into` → `to`: 10
- `their` → `ther`: 10
- `will` → `wil`: 10
- `and` → `ind`: 10
- `are` → `ar`: 10
- `it's` → `its`: 9
- `consumption` → `consemption`: 9
- `thee` → `the`: 9
- `of` → `o`: 8
- `oh` → `o`: 8
- `i'm` → `im`: 8
- `you` → `ou`: 8
- `this` → `thes`: 8
- `characteristic` → `caracteristic`: 7
- `wholly` → `holy`: 7
- `this` → `the`: 7
- `governor` → `govener`: 7
- `all` → `al`: 7
- `uncas` → `uncus`: 7
- `where` → `were`: 7

Top deleted words:

- `a`: 31
- `in`: 13
- `i`: 13
- `the`: 12
- `it`: 7
- `and`: 7
- `to`: 7
- `more`: 6
- `he`: 6
- `free`: 6
- `you`: 5
- `all`: 5
- `on`: 5
- `is`: 4
- `up`: 4
- `why`: 4
- `but`: 4
- `her`: 4
- `or`: 4
- `come`: 4
- `an`: 4
- `so`: 4
- `there`: 3
- `how`: 3
- `not`: 3
- `no`: 3
- `do`: 3
- `i've`: 3
- `mister`: 3
- `this`: 3

Top inserted words:

- `a`: 36
- `in`: 20
- `i`: 14
- `s`: 9
- `some`: 9
- `every`: 9
- `any`: 9
- `the`: 9
- `an`: 8
- `with`: 6
- `over`: 6
- `he`: 6
- `n`: 5
- `at`: 5
- `al`: 5
- `o`: 5
- `mont`: 5
- `be`: 4
- `up`: 4
- `to`: 4
- `what`: 4
- `fire`: 4
- `is`: 4
- `out`: 4
- `ad`: 4
- `man`: 4
- `main`: 4
- `never`: 4
- `there`: 3
- `had`: 3

Duration-bucket WER:

- 10-15s: 0.1653
- <5s: 0.3266
- 5-10s: 0.1942
- >=15s: 0.1601

Highest speaker-level WER:

- 1995: 0.2911
- 3570: 0.2768
- 8555: 0.2697
- 2961: 0.2595
- 4507: 0.2562
- 7176: 0.2553
- 7729: 0.2502
- 8463: 0.2435
- 4992: 0.2415
- 4077: 0.2307
- 2094: 0.2215
- 260: 0.2207
- 3575: 0.2149
- 6829: 0.2127
- 2300: 0.2111
- 4446: 0.2085
- 5142: 0.2048
- 908: 0.2031
- 4970: 0.2000
- 2830: 0.1991
- 8455: 0.1984
- 1188: 0.1975
- 61: 0.1951
- 8224: 0.1945
- 121: 0.1886
- 6930: 0.1873
- 1089: 0.1860
- 1284: 0.1845
- 5683: 0.1828
- 237: 0.1827

Representative high-error utterances:

- `1089-134691-0024` ref: stephanos dedalos / hyp: s tdaft firn nows deatd laos
- `8555-292519-0002` ref: venice / hyp: i a is
- `1089-134691-0003` ref: the university / hyp: thi yun of rcety
- `3575-170457-0016` ref: farewell madam / hyp: fear bone he tom
- `5105-28241-0014` ref: another circumstance was most remarkable / hyp: ind hotheseicon s ant ous meos hr mockable

## wav2vec2_masking_finetune_10h

Top substitutions:

- `in` → `and`: 28
- `and` → `in`: 23
- `a` → `the`: 15
- `and` → `an`: 14
- `o` → `oh`: 8
- `in` → `an`: 7
- `the` → `a`: 7
- `harry` → `herry`: 6
- `mary` → `marry`: 6
- `he's` → `is`: 6
- `wrote` → `rote`: 6
- `in` → `ind`: 6
- `it's` → `its`: 5
- `pepper` → `peper`: 5
- `add` → `ad`: 5
- `salt` → `sault`: 5
- `this` → `the`: 5
- `gwynplaine` → `gwenplain`: 5
- `criss` → `christ`: 5
- `it` → `at`: 4
- `bartley` → `bartly`: 4
- `and` → `ind`: 4
- `ingredients` → `ingredience`: 4
- `flour` → `flower`: 4
- `anyone` → `one`: 4
- `is` → `as`: 4
- `gryce` → `grice`: 4
- `clarke` → `clark`: 4
- `macklewain` → `macklewaine`: 4
- `honor` → `honour`: 4

Top deleted words:

- `a`: 24
- `and`: 10
- `to`: 8
- `the`: 5
- `of`: 4
- `are`: 4
- `in`: 4
- `don`: 4
- `red`: 3
- `us`: 3
- `an`: 3
- `had`: 3
- `over`: 3
- `mary`: 3
- `any`: 3
- `no`: 2
- `there`: 2
- `mac`: 2
- `it`: 2
- `or`: 2
- `some`: 2
- `new`: 2
- `i`: 2
- `every`: 2
- `gas`: 2
- `for`: 2
- `you`: 2
- `re`: 2
- `north`: 2
- `on`: 2

Top inserted words:

- `a`: 27
- `the`: 8
- `any`: 8
- `he`: 7
- `in`: 6
- `some`: 4
- `with`: 4
- `there`: 4
- `to`: 3
- `force`: 3
- `sweet`: 3
- `an`: 3
- `table`: 3
- `and`: 2
- `up`: 2
- `fire`: 2
- `i`: 2
- `sauce`: 2
- `which`: 2
- `down`: 2
- `de`: 2
- `where`: 2
- `may`: 2
- `man`: 2
- `t`: 2
- `dora`: 2
- `ever`: 2
- `arrouse`: 2
- `it`: 2
- `of`: 2

Duration-bucket WER:

- 5-10s: 0.1041
- <5s: 0.1229
- 10-15s: 0.1085

Highest speaker-level WER:

- 652: 0.2011
- 777: 0.1600
- 1272: 0.1562
- 1919: 0.1499
- 6313: 0.1439
- 1673: 0.1402
- 3576: 0.1400
- 8842: 0.1324
- 5694: 0.1265
- 251: 0.1258
- 1988: 0.1238
- 3752: 0.1232
- 2803: 0.1185
- 2078: 0.1168
- 3000: 0.1135
- 84: 0.1105
- 7850: 0.1079
- 6295: 0.1077
- 6241: 0.1065
- 5895: 0.1063
- 5536: 0.1059
- 2086: 0.1058
- 1462: 0.1053
- 2902: 0.1047
- 5338: 0.1047
- 2035: 0.1040
- 174: 0.1000
- 3853: 0.0944
- 1993: 0.0917
- 8297: 0.0902

Representative high-error utterances:

- `8842-304647-0001` ref: quinci impara a stupirti / hyp: quein shi empauras to beiertiy
- `1919-142785-0048` ref: french forcemeat / hyp: french force meet
- `1919-142785-0056` ref: fried bread crumbs / hyp: fride breadcrums
- `2078-142845-0000` ref: kirkleatham yeast / hyp: cirklytheam asted
- `2078-142845-0044` ref: italian rusks / hyp: etalian rusqks

## wav2vec2_masking_finetune_10h

Top substitutions:

- `and` → `in`: 26
- `in` → `and`: 17
- `and` → `an`: 17
- `a` → `the`: 14
- `in` → `an`: 11
- `the` → `a`: 11
- `an` → `and`: 11
- `is` → `as`: 10
- `robin` → `robbin`: 10
- `paul` → `pal`: 8
- `o` → `oh`: 7
- `this` → `the`: 7
- `knife` → `nife`: 7
- `sought` → `saught`: 6
- `shone` → `shown`: 6
- `anyone` → `one`: 6
- `edison` → `eddison`: 6
- `system` → `sistom`: 6
- `leisure` → `leasure`: 6
- `in` → `ind`: 5
- `to` → `too`: 5
- `taylor` → `tailer`: 5
- `cotton` → `cotten`: 5
- `mary` → `marry`: 5
- `the` → `te`: 5
- `christ` → `crist`: 5
- `silvia` → `sylvia`: 5
- `slang` → `slaying`: 5
- `were` → `where`: 5
- `servadac` → `servadack`: 5

Top deleted words:

- `a`: 25
- `and`: 12
- `the`: 8
- `or`: 8
- `to`: 7
- `in`: 6
- `on`: 4
- `more`: 4
- `latter`: 4
- `new`: 4
- `you`: 3
- `this`: 3
- `he`: 3
- `for`: 3
- `have`: 3
- `i`: 3
- `it`: 3
- `will`: 3
- `ben`: 3
- `la`: 3
- `mademoiselle`: 3
- `school`: 2
- `all`: 2
- `of`: 2
- `admit`: 2
- `is`: 2
- `up`: 2
- `then`: 2
- `utility`: 2
- `we`: 2

Top inserted words:

- `a`: 17
- `any`: 11
- `in`: 11
- `i`: 10
- `the`: 9
- `there`: 8
- `to`: 6
- `up`: 5
- `o`: 5
- `some`: 5
- `and`: 5
- `every`: 5
- `he`: 5
- `never`: 5
- `be`: 4
- `for`: 4
- `of`: 4
- `t`: 3
- `what`: 3
- `with`: 3
- `forth`: 3
- `sun`: 3
- `you`: 3
- `it`: 3
- `an`: 3
- `here`: 3
- `main`: 3
- `mount`: 3
- `top`: 3
- `play`: 3

Duration-bucket WER:

- 10-15s: 0.1085
- <5s: 0.1300
- 5-10s: 0.1110
- >=15s: 0.1086

Highest speaker-level WER:

- 7176: 0.1706
- 8555: 0.1642
- 3570: 0.1546
- 2961: 0.1531
- 4992: 0.1516
- 7729: 0.1512
- 1995: 0.1495
- 2300: 0.1358
- 61: 0.1337
- 4077: 0.1335
- 8463: 0.1306
- 260: 0.1236
- 1089: 0.1195
- 2830: 0.1190
- 4507: 0.1187
- 6829: 0.1182
- 2094: 0.1171
- 4970: 0.1154
- 908: 0.1144
- 5142: 0.1138
- 5105: 0.1074
- 4446: 0.1065
- 1284: 0.1024
- 1188: 0.0995
- 237: 0.0993
- 6930: 0.0991
- 8455: 0.0981
- 5639: 0.0975
- 5683: 0.0974
- 121: 0.0970

Representative high-error utterances:

- `1089-134691-0024` ref: stephanos dedalos / hyp: steffinels dead loss
- `3575-170457-0016` ref: farewell madam / hyp: fear well madame
- `2830-3980-0026` ref: verse two / hyp: first tube
- `61-70968-0038` ref: robin fitzooth / hyp: robein fitsuf
- `672-122797-0049` ref: squeak squeak / hyp: squiak squick

## wav2vec2_top3_finetune_10h

Top substitutions:

- `in` → `and`: 23
- `quite` → `quit`: 17
- `know` → `now`: 15
- `and` → `in`: 15
- `here` → `heare`: 14
- `and` → `an`: 14
- `until` → `untill`: 14
- `for` → `fore`: 13
- `been` → `ben`: 13
- `their` → `theire`: 13
- `thee` → `the`: 13
- `randal` → `randle`: 13
- `into` → `to`: 12
- `know` → `kno`: 11
- `it's` → `its`: 11
- `the` → `th`: 10
- `white` → `whit`: 10
- `too` → `to`: 10
- `far` → `fare`: 10
- `nor` → `nore`: 10
- `in` → `an`: 9
- `too` → `two`: 9
- `add` → `ad`: 9
- `altogether` → `together`: 9
- `a` → `the`: 9
- `either` → `ither`: 9
- `no` → `kno`: 9
- `soul` → `sole`: 9
- `gone` → `gon`: 8
- `guard` → `gard`: 8

Top deleted words:

- `a`: 25
- `in`: 7
- `with`: 7
- `the`: 6
- `and`: 5
- `to`: 4
- `i`: 4
- `any`: 4
- `don`: 4
- `of`: 3
- `it`: 3
- `illustration`: 3
- `are`: 3
- `but`: 3
- `some`: 3
- `vita`: 3
- `they`: 2
- `no`: 2
- `mac`: 2
- `t`: 2
- `this`: 2
- `good`: 2
- `new`: 2
- `red`: 2
- `looked`: 2
- `other`: 2
- `mary`: 2
- `told`: 2
- `him`: 2
- `em`: 2

Top inserted words:

- `a`: 37
- `in`: 19
- `any`: 16
- `over`: 11
- `with`: 10
- `the`: 9
- `un`: 9
- `all`: 9
- `some`: 9
- `i`: 8
- `out`: 7
- `an`: 7
- `he`: 6
- `what`: 6
- `there`: 6
- `to`: 5
- `my`: 5
- `grand`: 5
- `mean`: 5
- `be`: 5
- `mag`: 5
- `al`: 4
- `o`: 4
- `every`: 4
- `ad`: 4
- `who`: 4
- `it`: 4
- `no`: 4
- `that`: 3
- `its`: 3

Duration-bucket WER:

- 5-10s: 0.2112
- <5s: 0.2481
- 10-15s: 0.2099

Highest speaker-level WER:

- 652: 0.2942
- 777: 0.2869
- 1673: 0.2804
- 251: 0.2631
- 6313: 0.2584
- 1988: 0.2564
- 1272: 0.2517
- 8842: 0.2491
- 1919: 0.2485
- 3752: 0.2449
- 2078: 0.2384
- 5694: 0.2379
- 84: 0.2312
- 3081: 0.2192
- 5338: 0.2182
- 3576: 0.2177
- 5895: 0.2170
- 6295: 0.2162
- 2902: 0.2112
- 7850: 0.2104
- 422: 0.2098
- 174: 0.2075
- 6241: 0.2072
- 2035: 0.2072
- 2803: 0.2065
- 2086: 0.2050
- 1462: 0.2028
- 5536: 0.2010
- 8297: 0.2009
- 1993: 0.1948

Representative high-error utterances:

- `2428-83705-0036` ref: someone sniggered / hyp: some one sniggeredaiaoo
- `5895-34615-0005` ref: gwynplaine was a mountebank / hyp: guin plain waus a mount o bank
- `5895-34615-0011` ref: an everlasting laugh / hyp: and ever lasting lauf
- `6313-76958-0008` ref: humph grunted curley adams / hyp: hh f groanted curly adoms
- `1272-135031-0019` ref: that's funny remarked betsy thoughtfully / hyp: that 's funy remarked betcy thought fuly

## wav2vec2_top3_finetune_10h

Top substitutions:

- `thee` → `the`: 29
- `quite` → `quit`: 21
- `and` → `in`: 21
- `and` → `an`: 19
- `christ` → `crist`: 18
- `their` → `theire`: 18
- `been` → `ben`: 17
- `know` → `kno`: 17
- `great` → `greate`: 16
- `air` → `aire`: 16
- `far` → `fare`: 15
- `in` → `and`: 14
- `in` → `an`: 14
- `for` → `fore`: 14
- `will` → `wil`: 13
- `no` → `kno`: 13
- `know` → `now`: 13
- `an` → `and`: 13
- `paul` → `pall`: 12
- `bartley` → `bartly`: 12
- `hear` → `heare`: 11
- `red` → `read`: 11
- `a` → `the`: 11
- `window` → `windo`: 11
- `oh` → `o`: 11
- `i'm` → `im`: 11
- `captain` → `captin`: 11
- `doubt` → `dout`: 10
- `are` → `ar`: 10
- `there's` → `theres`: 10

Top deleted words:

- `a`: 18
- `the`: 11
- `and`: 9
- `in`: 8
- `or`: 8
- `more`: 6
- `you`: 5
- `it`: 5
- `with`: 4
- `at`: 4
- `la`: 4
- `are`: 3
- `new`: 3
- `we`: 3
- `this`: 3
- `he`: 3
- `could`: 3
- `e`: 3
- `latter`: 3
- `do`: 3
- `ben`: 3
- `fir`: 3
- `mademoiselle`: 3
- `have`: 3
- `through`: 2
- `god`: 2
- `other`: 2
- `of`: 2
- `don't`: 2
- `education`: 2

Top inserted words:

- `a`: 34
- `in`: 16
- `any`: 15
- `un`: 13
- `some`: 12
- `over`: 12
- `i`: 12
- `with`: 11
- `the`: 11
- `every`: 10
- `to`: 9
- `all`: 9
- `there`: 8
- `what`: 8
- `mean`: 6
- `no`: 6
- `he`: 6
- `up`: 5
- `man`: 5
- `an`: 5
- `under`: 5
- `re`: 5
- `where`: 4
- `child`: 4
- `is`: 4
- `fire`: 4
- `out`: 4
- `it`: 4
- `you`: 4
- `her`: 4

Duration-bucket WER:

- 10-15s: 0.2164
- <5s: 0.2540
- 5-10s: 0.2168
- >=15s: 0.2180

Highest speaker-level WER:

- 3570: 0.2944
- 1995: 0.2825
- 7729: 0.2798
- 8555: 0.2697
- 4077: 0.2677
- 2961: 0.2638
- 8463: 0.2581
- 7176: 0.2512
- 260: 0.2504
- 6829: 0.2492
- 4992: 0.2474
- 2830: 0.2395
- 2300: 0.2365
- 4446: 0.2353
- 4970: 0.2337
- 908: 0.2324
- 5142: 0.2299
- 2094: 0.2282
- 61: 0.2262
- 3575: 0.2235
- 5683: 0.2230
- 4507: 0.2219
- 5105: 0.2210
- 7127: 0.2190
- 8455: 0.2064
- 121: 0.2055
- 1284: 0.2054
- 8224: 0.2043
- 1089: 0.2029
- 237: 0.2007

Representative high-error utterances:

- `1089-134691-0024` ref: stephanos dedalos / hyp: steffinous dead los
- `3575-170457-0016` ref: farewell madam / hyp: faire well madame
- `260-123288-0014` ref: hans stirs not / hyp: honds star s notied
- `121-123852-0001` ref: ay me / hyp: y m
- `121-127105-0014` ref: you are acute / hyp: youo are a cut

## wav2vec2_top6_finetune_10h

Top substitutions:

- `in` → `and`: 29
- `and` → `in`: 21
- `and` → `an`: 15
- `the` → `a`: 11
- `until` → `untill`: 11
- `in` → `an`: 10
- `mary` → `marry`: 10
- `a` → `the`: 9
- `o` → `oh`: 8
- `add` → `ad`: 8
- `eggs` → `egs`: 8
- `night` → `knight`: 7
- `randal` → `randl`: 7
- `pepper` → `peper`: 6
- `eye` → `ey`: 6
- `divided` → `devided`: 6
- `to` → `too`: 5
- `ingredients` → `ingredience`: 5
- `occurred` → `occured`: 5
- `carrie` → `carry`: 5
- `wrote` → `rote`: 5
- `randal` → `randle`: 5
- `bartley` → `bartly`: 4
- `europe` → `europ`: 4
- `cosette` → `cosete`: 4
- `in` → `ind`: 4
- `egg` → `eg`: 4
- `flour` → `flower`: 4
- `wholly` → `holy`: 4
- `christmas` → `cristmas`: 4

Top deleted words:

- `a`: 25
- `the`: 13
- `and`: 9
- `to`: 6
- `in`: 6
- `it`: 5
- `are`: 5
- `with`: 5
- `don`: 5
- `some`: 4
- `i`: 4
- `any`: 4
- `mary`: 4
- `of`: 3
- `on`: 3
- `for`: 3
- `vita`: 3
- `you`: 2
- `us`: 2
- `so`: 2
- `illustration`: 2
- `serve`: 2
- `have`: 2
- `you're`: 2
- `there`: 2
- `dead`: 2
- `every`: 2
- `that`: 2
- `em`: 2
- `find`: 2

Top inserted words:

- `a`: 27
- `the`: 10
- `in`: 8
- `any`: 7
- `grand`: 5
- `sweet`: 5
- `of`: 4
- `i`: 4
- `down`: 4
- `fire`: 3
- `he`: 3
- `may`: 3
- `who`: 3
- `ever`: 3
- `there`: 3
- `every`: 3
- `out`: 3
- `table`: 3
- `up`: 2
- `it`: 2
- `garro`: 2
- `degaro`: 2
- `with`: 2
- `chest`: 2
- `near`: 2
- `man`: 2
- `wh`: 2
- `water`: 2
- `had`: 2
- `at`: 2

Duration-bucket WER:

- 5-10s: 0.1150
- <5s: 0.1303
- 10-15s: 0.1172

Highest speaker-level WER:

- 652: 0.2058
- 1919: 0.1694
- 777: 0.1614
- 1272: 0.1562
- 1673: 0.1541
- 6313: 0.1483
- 8842: 0.1436
- 3576: 0.1416
- 251: 0.1411
- 5694: 0.1328
- 1988: 0.1310
- 3752: 0.1264
- 2086: 0.1258
- 84: 0.1246
- 2803: 0.1242
- 7850: 0.1205
- 2902: 0.1204
- 3000: 0.1198
- 5895: 0.1187
- 6241: 0.1178
- 2078: 0.1152
- 3081: 0.1113
- 5536: 0.1108
- 8297: 0.1107
- 5338: 0.1096
- 2035: 0.1085
- 174: 0.1075
- 7976: 0.1033
- 2428: 0.1014
- 422: 0.1000

Representative high-error utterances:

- `8842-304647-0001` ref: quinci impara a stupirti / hyp: quen she emparas to beartiy
- `1919-142785-0006` ref: illustration the cucumber / hyp: illoustration the cew comber
- `1919-142785-0024` ref: illustration ginger / hyp: illustrationjinjer
- `1919-142785-0063` ref: illustration sage / hyp: illustrationsage
- `2078-142845-0000` ref: kirkleatham yeast / hyp: ceurkleytham yuast

## wav2vec2_top6_finetune_10h

Top substitutions:

- `and` → `in`: 32
- `in` → `and`: 20
- `christ` → `crist`: 18
- `and` → `an`: 18
- `a` → `the`: 13
- `in` → `an`: 10
- `an` → `and`: 10
- `paul` → `pall`: 10
- `o` → `oh`: 8
- `the` → `a`: 8
- `until` → `untill`: 8
- `wholly` → `holy`: 6
- `leisure` → `leasure`: 6
- `this` → `the`: 6
- `taylor` → `tailer`: 6
- `mary` → `marry`: 6
- `the` → `he`: 6
- `phronsie` → `fronzy`: 6
- `opinion` → `oppinion`: 5
- `eye` → `ey`: 5
- `shone` → `shown`: 5
- `uncas` → `uncus`: 5
- `anyone` → `one`: 5
- `holmes` → `homes`: 5
- `the` → `te`: 5
- `system` → `sistom`: 5
- `paul` → `pal`: 5
- `is` → `his`: 5
- `your` → `you`: 5
- `writing` → `riting`: 5

Top deleted words:

- `a`: 29
- `the`: 11
- `and`: 9
- `in`: 7
- `or`: 6
- `i`: 5
- `new`: 5
- `to`: 4
- `it`: 3
- `up`: 3
- `this`: 3
- `more`: 3
- `ben`: 3
- `on`: 3
- `school`: 2
- `an`: 2
- `sick`: 2
- `you`: 2
- `admit`: 2
- `with`: 2
- `commentary`: 2
- `he`: 2
- `place`: 2
- `through`: 2
- `why`: 2
- `any`: 2
- `do`: 2
- `de`: 2
- `not`: 2
- `there`: 2

Top inserted words:

- `a`: 18
- `in`: 10
- `the`: 10
- `i`: 9
- `any`: 8
- `every`: 8
- `up`: 6
- `there`: 6
- `he`: 6
- `never`: 6
- `to`: 5
- `for`: 5
- `mount`: 5
- `be`: 4
- `some`: 4
- `fire`: 4
- `with`: 4
- `where`: 3
- `house`: 3
- `is`: 3
- `forth`: 3
- `an`: 3
- `and`: 3
- `top`: 3
- `play`: 3
- `on`: 2
- `under`: 2
- `tend`: 2
- `what`: 2
- `de`: 2

Duration-bucket WER:

- 10-15s: 0.1197
- <5s: 0.1327
- 5-10s: 0.1183
- >=15s: 0.1183

Highest speaker-level WER:

- 7176: 0.1713
- 3570: 0.1695
- 8555: 0.1605
- 1995: 0.1604
- 7729: 0.1564
- 2961: 0.1548
- 8463: 0.1500
- 61: 0.1458
- 4992: 0.1426
- 4077: 0.1420
- 2830: 0.1414
- 260: 0.1401
- 2300: 0.1334
- 1089: 0.1323
- 4507: 0.1281
- 4970: 0.1273
- 5142: 0.1251
- 2094: 0.1245
- 6829: 0.1229
- 3575: 0.1218
- 908: 0.1217
- 5105: 0.1167
- 5639: 0.1166
- 1188: 0.1127
- 4446: 0.1092
- 6930: 0.1084
- 8224: 0.1056
- 237: 0.1014
- 1221: 0.1011
- 8455: 0.1010

Representative high-error utterances:

- `1089-134691-0024` ref: stephanos dedalos / hyp: staffenos dead los
- `3575-170457-0016` ref: farewell madam / hyp: fair well madame
- `61-70968-0038` ref: robin fitzooth / hyp: robpein fits oouth
- `2830-3980-0026` ref: verse two / hyp: first toob
- `8555-292519-0002` ref: venice / hyp: enat

## wavlm_finetune_10h

Top substitutions:

- `in` → `and`: 32
- `and` → `in`: 30
- `in` → `an`: 19
- `and` → `an`: 17
- `and` → `ind`: 11
- `a` → `the`: 10
- `the` → `a`: 10
- `an` → `and`: 10
- `in` → `ind`: 9
- `thee` → `the`: 9
- `bartley` → `bartly`: 7
- `mary` → `marry`: 7
- `randal` → `randel`: 7
- `it's` → `its`: 6
- `as` → `is`: 6
- `o` → `oh`: 6
- `that` → `the`: 6
- `add` → `ad`: 6
- `a` → `of`: 6
- `bread` → `bred`: 6
- `led` → `let`: 5
- `of` → `a`: 5
- `pepper` → `peper`: 5
- `flour` → `flower`: 5
- `is` → `his`: 5
- `phoebe` → `feby`: 5
- `carrie` → `carry`: 5
- `a` → `to`: 5
- `you're` → `your`: 5
- `harry` → `herry`: 5

Top deleted words:

- `a`: 41
- `to`: 14
- `in`: 13
- `and`: 11
- `of`: 6
- `any`: 6
- `the`: 6
- `on`: 5
- `do`: 5
- `it`: 5
- `some`: 5
- `i`: 5
- `no`: 4
- `red`: 3
- `they`: 3
- `mary`: 3
- `an`: 3
- `can`: 3
- `as`: 3
- `how`: 2
- `new`: 2
- `close`: 2
- `each`: 2
- `apple`: 2
- `there`: 2
- `flower`: 2
- `serve`: 2
- `are`: 2
- `wagon`: 2
- `take`: 2

Top inserted words:

- `a`: 33
- `the`: 13
- `in`: 12
- `to`: 10
- `it`: 6
- `any`: 6
- `i`: 5
- `s`: 5
- `there`: 5
- `he`: 4
- `with`: 4
- `t`: 4
- `mean`: 4
- `for`: 3
- `up`: 3
- `fire`: 3
- `force`: 3
- `some`: 3
- `foot`: 3
- `of`: 3
- `my`: 3
- `thi`: 3
- `blue`: 3
- `and`: 3
- `may`: 3
- `table`: 3
- `e`: 2
- `v`: 2
- `al`: 2
- `too`: 2

Duration-bucket WER:

- 5-10s: 0.1582
- <5s: 0.1846
- 10-15s: 0.1637

Highest speaker-level WER:

- 6313: 0.2403
- 1272: 0.2398
- 777: 0.2389
- 652: 0.2350
- 1988: 0.2074
- 5694: 0.1897
- 8842: 0.1894
- 5536: 0.1886
- 1673: 0.1866
- 1919: 0.1858
- 251: 0.1830
- 2803: 0.1817
- 5338: 0.1771
- 3576: 0.1735
- 3752: 0.1730
- 7850: 0.1709
- 5895: 0.1683
- 2078: 0.1680
- 174: 0.1600
- 6241: 0.1590
- 2035: 0.1552
- 84: 0.1544
- 2902: 0.1501
- 6319: 0.1501
- 3853: 0.1489
- 6295: 0.1487
- 2412: 0.1487
- 8297: 0.1470
- 1462: 0.1466
- 1993: 0.1461

Representative high-error utterances:

- `2078-142845-0000` ref: kirkleatham yeast / hyp: coirkly them yeastd
- `5895-34615-0005` ref: gwynplaine was a mountebank / hyp: guin plain was the mount to bank
- `8842-304647-0001` ref: quinci impara a stupirti / hyp: queenche emparr aus tu berty
- `1272-135031-0019` ref: that's funny remarked betsy thoughtfully / hyp: that's funey remarked a betse thought fully
- `1919-142785-0048` ref: french forcemeat / hyp: french force meeat
