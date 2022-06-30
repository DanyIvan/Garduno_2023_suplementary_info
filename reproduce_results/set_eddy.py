import numpy as np

def set_eddy_diff(model, h=None, temp=None, keep_up_constant=False, 
    use_kevin_formula=False):
    '''Change eddy diffusivity profile to match a tropopause defined by height or
    by temperature.'''
    eddy = model.vars.edd
    press = model.vars.press
    diff_eddy=np.diff(eddy)
    step = np.where(diff_eddy != diff_eddy[0])[0][0] + 1
    if temp:
        h_temp_idx = np.where(np.array(model.vars.t) <= temp)[0][0]
        h_temp = model.data.z[h_temp_idx]
    h = h if h else h_temp
    model.data.ztrop = h
    z = np.where(model.data.z >= h)[0][0]
    model.data.jtrop = z + 1
    if use_kevin_formula:
        eddy[:z] = 1e5
        eddy[z:] = 1e3 * np.sqrt(press[z] / press[z:])
        model.vars.edd[:] = eddy
    else:
        if z > step:
            insert = [eddy[0]] * (z - step)
            if keep_up_constant:
                eddy = np.concatenate([eddy[:step], insert, eddy[z:]]) 
            else:
                eddy = np.concatenate([insert, eddy[:-len(insert)]]) 
            model.vars.edd[:] = eddy
        else:
            eddy = np.concatenate([eddy[step - z:], [eddy[-1]] * (step - z)]) 
            model.vars.edd[:] = eddy