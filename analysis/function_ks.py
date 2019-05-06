import numpy as np
import pandas as pd


def calc_ks(nd_fcc, rd_fcc, nd_bcc, rd_bcc, comb_array, norm_array):
    total_array = np.hstack((comb_array, norm_array))
    # ===============================================================================
    # calculation bcc to fcc transformation matrix A
    # AxB=F
    # A=FxB_inverse
    # ===============================================================================
    # TD = ND X RD
    td_fcc = np.cross(nd_fcc, rd_fcc)
    td_bcc = np.cross(nd_bcc, rd_bcc)
    # fcc_norm
    nd_fcc_ = nd_fcc / np.linalg.norm(nd_fcc)
    rd_fcc_ = rd_fcc / np.linalg.norm(rd_fcc)
    td_fcc_ = td_fcc / np.linalg.norm(td_fcc)
    # bcc_norm
    nd_bcc_ = nd_bcc / np.linalg.norm(nd_bcc)
    rd_bcc_ = rd_bcc / np.linalg.norm(rd_bcc)
    td_bcc_ = td_bcc / np.linalg.norm(td_bcc)
    # F_matrix, B_matrix
    f_matrix = np.array([nd_fcc_, rd_fcc_, td_fcc_]).T
    b_matrix = np.array([nd_bcc_, rd_bcc_, td_bcc_]).T
    # B_inverse
    b_inverse = np.linalg.inv(b_matrix)
    # A = F × B_inverse
    a_matrix = np.dot(f_matrix, b_inverse)
    # ===============================================================================
    # calculation deviation of plane parallel and direction parallel
    # ===============================================================================

    p_fcc = total_array[:, 12:15]
    p_bcc = total_array[:, 18:21]
    d_fcc = total_array[:, 15:18]
    d_bcc = total_array[:, 21:24]

    p_bcc_trans = np.dot(a_matrix, p_bcc.T)
    d_bcc_trans = np.dot(a_matrix, d_bcc.T)

    p_deviation = []
    d_deviation = []
    ks_deviation = []
    rotation_axis = []
    for i in range(0, total_array.shape[0]):
        # plane
        bcc_p = p_bcc_trans[:, i]
        fcc_p = p_fcc[i, :]
        p_cos = np.dot(bcc_p, fcc_p) / (np.linalg.norm(bcc_p) * np.linalg.norm(fcc_p))
        p_theta_rad = np.arccos(p_cos)
        p_theta_deg = np.rad2deg(p_theta_rad)
        p_deviation.append([p_theta_deg])
        # direction
        bcc_d = d_bcc_trans[:, i]
        fcc_d = d_fcc[i, :]
        d_cos = np.dot(bcc_d, fcc_d) / (np.linalg.norm(bcc_d) * np.linalg.norm(fcc_d))
        d_theta_rad = np.arccos(d_cos)
        d_theta_deg = np.rad2deg(d_theta_rad)
        d_deviation.append([d_theta_deg])
        # K-S
        # AA x BB = FF
        fcc_t = np.cross(fcc_p, fcc_d)
        ff_matrix = np.array([fcc_p, fcc_d, fcc_t]).T
        bcc_pp = p_bcc[i, :]
        bcc_dd = d_bcc[i, :]
        bcc_t = np.cross(bcc_pp, bcc_dd)
        bb_matrix = np.array([bcc_pp, bcc_dd, bcc_t]).T
        # AA = FFxBB_inverse
        bb_inverse = np.linalg.inv(bb_matrix)
        aa_matrix = np.dot(ff_matrix, bb_inverse)
        # AA x deltaA = A
        # deltaA = A x AA_inverse
        aa_inverse = np.linalg.inv(aa_matrix)
        delta_a = np.dot(a_matrix, aa_inverse)
        # Rodrigues' rotation formula
        # k = n x sin
        kx = (delta_a[2, 1] - delta_a[1, 2]) / 2
        ky = (delta_a[0, 2] - delta_a[2, 0]) / 2
        kz = (delta_a[1, 0] - delta_a[0, 1]) / 2
        sin = np.linalg.norm([kx, ky, kz])
        ks_theta_rad = np.arcsin(sin)
        ks_theta_deg = np.rad2deg(ks_theta_rad)
        ks_deviation.append([ks_theta_deg])
        # rotation axis (n)
        ax = [kx, ky, kz] / np.linalg.norm([kx, ky, kz])
        rotation_axis.append(ax)

    p_array = np.array(p_deviation)
    d_array = np.array(d_deviation)
    ks_array = np.array(ks_deviation)
    ax_array = np.array(rotation_axis)

    result_array = np.hstack([comb_array, p_array, d_array, ks_array, ax_array])
    cols = ["fcc_p_h", "fcc_p_k", "fcc_p_l", "fcc_d_h", "fcc_d_k", "fcc_d_l",
            "bcc_p_h", "bcc_p_k", "bcc_p_", "bcc_d_h", "bcc_d_k", "bcc_d_l",
            "plane", "direction", "K-S", "ax_x", "ax_y", "ax_z"]
    df_results = pd.DataFrame(result_array, columns=cols)

    df_results["K-S"] = df_results["K-S"].where((df_results["K-S"] > df_results["plane"]), np.nan)
    df_results["K-S"] = df_results["K-S"].where((df_results["K-S"] > df_results["direction"]), np.nan)
    df_results = df_results.dropna()
    df_results = df_results.sort_values("K-S")

    fcc_plane = "(" + str(int(df_results.iloc[0, 0])) + str(int(df_results.iloc[0, 1])) + str(
        int(df_results.iloc[0, 2])) + ")"
    fcc_direction = "(" + str(int(df_results.iloc[0, 3])) + str(int(df_results.iloc[0, 4])) + str(
        int(df_results.iloc[0, 5])) + ")"
    bcc_plane = "(" + str(int(df_results.iloc[0, 6])) + str(int(df_results.iloc[0, 7])) + str(
        int(df_results.iloc[0, 8])) + ")"
    bcc_direction = "(" + str(int(df_results.iloc[0, 9])) + str(int(df_results.iloc[0, 10])) + str(
        int(df_results.iloc[0, 11])) + ")"

    plane = df_results.iloc[0, 12]
    direction = df_results.iloc[0, 13]
    ks = df_results.iloc[0, 14]
    result_list = [fcc_plane, fcc_direction, bcc_plane, bcc_direction, plane, direction, ks]
    return result_list
