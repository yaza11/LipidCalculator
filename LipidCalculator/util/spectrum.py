from itertools import product


def recursively_merge_peaks_to_resolution(mzs, intensities, mz_tol: int = None, resolution: int = None):
    assert (tol_is_abs := (mz_tol is not None)) ^ (resolution is not None)
    all_above_tol = False

    intensities = list(intensities)
    mzs = list(mzs)

    while not all_above_tol:
        # look for mass pair below mass tolerance
        for (idx1, (i1, mz1)), (idx2, (i2, mz2)) in product(
                enumerate(zip(intensities, mzs)),
                enumerate(zip(intensities, mzs))
        ):
            if idx1 == idx2:
                continue
            dmz = abs(mz1 - mz2)
            if tol_is_abs:
                if dmz > mz_tol:
                    continue
            # merge if R < m / dm ==> dm < m / R
            elif dmz > (mz1 / 2 + mz2 / 2) / resolution:
                continue
            # found close mz values
            # merge mz and intensity values
            mz_new = (mz1 * i1 + mz2 * i2) / (i1 + i2)
            i_new = i1 + i2

            # pop the bigger index first
            for idx in sorted([idx1, idx2], reverse=True):
                intensities.pop(idx)
                mzs.pop(idx)
            # insert the merged values
            intensities.append(i_new)
            mzs.append(mz_new)
            break
        else:
            all_above_tol = True

    # sort by masses
    return zip(*sorted(zip(mzs, intensities)))
