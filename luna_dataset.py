#-*- coding:utf-8 -*-
#!/usr/bin/env python3

import os
import copy
from collections import namedtuple
import glob
import pandas as pd
import numpy as np
import functools
import SimpleITK as sitk


from utils import XyzTuple, xyz2irc, getCache

import torch
import torch.utils.data as data
from logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# CandidateListは辞書型変数
CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule_bool, diameter_mm, series_uid, center_xyz',
)

dataset_dir = '/home/take/fun/dataset/LUNA'

# キャッシュの設定
raw_cache = getCache('part2ch11_raw')

@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

# cacheを使う。
@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):


    # データセットのパスの一覧を取得
    mhd_path_list = glob.glob(dataset_dir + '/data/subset*/*.mhd')

    # ファイル名だけ抽出. 画像が存在するUIDリストを生成する
    # 名前でアクセスできるようにデータは集合で扱う
    # すべて記述すると以下のような書き方
    #mhd_list = set()
    #for mhd_path in mhd_path_list:
    #    mhd_list.add(os.path.split(mhd_path)[-1][:-4])
    mhd_list = {os.path.split(mhd_path)[-1][:-4] for mhd_path in mhd_path_list}

    # csv読み込み
    df_cand = pd.read_csv(dataset_dir + '/candidates.csv')
    df_anno = pd.read_csv(dataset_dir + '/annotations.csv')

    diameter_dict = {}
    # １つのUIDに対して複数のデータが存在しているため、
    # 辞書型の変数でデータを扱う。(key:UID, values:((x,y,z), diameter))
    for _, anno_row in df_anno.iterrows():
        diameter_dict.setdefault(anno_row.seriesuid, []).append(
                ((anno_row.coordX, anno_row.coordY, anno_row.coordZ), anno_row.diameter_mm) )

    candidateInfo_list = []
    for index, cand_row in df_cand.iterrows():
        # 画像が存在しなければスキップ
        if cand_row.seriesuid not in mhd_list and requireOnDisk_bool:
            print(index, 'skip')
            continue

        cand_diameter_mm = 0.0

        for anno_tup in diameter_dict.get(cand_row.seriesuid, []):
            anno_xyz, anno_diameter_mm = anno_tup
            anno_xyz_np = np.array((anno_xyz[0], anno_xyz[1], anno_xyz[2]))
            cand_xyz_np = np.array((cand_row.coordX, cand_row.coordY, cand_row.coordZ))
            if abs(cand_xyz_np - anno_xyz_np).all() > anno_diameter_mm / 4:
                cand_diameter_mm = 0.0
                break
            else:
                cand_diameter_mm = anno_diameter_mm
                break

        print(index, cand_row.seriesuid, cand_diameter_mm)

        # 出力生成
        candidateInfo_list.append(CandidateInfoTuple(
            bool(int(cand_row[4])),
            cand_diameter_mm,
            cand_row.seriesuid,
            (cand_row.coordX, cand_row.coordY, cand_row.coordZ) ))
    print('Finish creating candidate information', len(candidateInfo_list))
    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list


class Ct:
    def __init__(self, series_uid):
        mhd_path = glob.glob(dataset_dir + '/data//subset*/{}.mhd'.format(series_uid))[0]
        #if os.path.exists(str(mhd_path)):
        #    print('Do not exists mhd file')
        #    return

        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        #print(ct_a.shape)
        # clip
        ct_a.clip(-1000, 1000, ct_a)

        self.series_uid = series_uid
        self.hu_a = ct_a
        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = xyz2irc( center_xyz, self.origin_xyz,
                              self.vxSize_xyz, self.direction_a )
        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx   = int(start_ndx + width_irc[axis])
            if start_ndx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))
        ct_chunk = self.hu_a[tuple(slice_list)]
        return ct_chunk, center_irc


@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)


@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    return ct.getRawCandidate(center_xyz, width_irc)


class LunaDataset(data.Dataset):
    def __init__(self, val_stride=0, isValSet_bool=None, series_uid=None):
        self.candidateInfo_list = copy.copy(getCandidateInfoList())

        if series_uid:
            self.candidateInfo_list = [x for x in self.candidateInfo_list if x.series_uid == series_uid]

        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]

        elif val_stride > 0:
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list

        log.info("{!r}: {} {} samples".format( self, len(self.candidateInfo_list),
                                               "validation" if isValSet_bool else "training", ))

    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        candidateInfo_tup = self.candidateInfo_list[ndx]
        width_irc = (32, 48, 48)

        candidate_a, center_irc = getCtRawCandidate(
                                        candidateInfo_tup.series_uid,
                                        candidateInfo_tup.center_xyz,
                                        width_irc,)

        #print(candidate_a.shape)
        #print(torch.tensor(center_irc).shape)

        candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0)

        pos_t = torch.tensor([
                        not candidateInfo_tup.isNodule_bool,
                        candidateInfo_tup.isNodule_bool], dtype=torch.long)
        return candidate_t, pos_t, candidateInfo_tup.series_uid, torch.tensor(center_irc)


if __name__ == '__main__':
    candidateInfo_list = getCandidateInfoList(requireOnDisk_bool=False)
    #print(candidateInfo_list)
    positiveInfo_list = [x for x in candidateInfo_list if x[0]]
    diameter_list = [x[1] for x in positiveInfo_list]
    print(len(diameter_list))
    for i in range(0, len(diameter_list), 100):
        print('{:4} {:4.1f}mm'.format(i, diameter_list[i]))
    #print(LunaDataset()[0])
