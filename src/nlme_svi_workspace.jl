# A unit's layout stores only its local coordinate permutation and positions
# in sorted [globals; locals]. Never replicate the globals for every unit.
function _brm_svi_unit_layout(globals, block)
    selection = _brm_nlme_unit_selection(globals, block)
    (; selection.order, selection.slots)
end

function _brm_svi_joint!(values, layout, g, z)
    previous, global_index = 0, 1
    for (k, slot) in enumerate(layout.slots)
        for j in previous+1:slot-1
            values[j] = g[global_index]
            global_index += 1
        end
        values[slot] = z[layout.order[k]]
        previous = slot
    end
    for j in previous+1:length(values)
        values[j] = g[global_index]
        global_index += 1
    end
    values
end

function _brm_svi_split_gradient!(gg, gz, layout, gradient)
    previous, global_index = 0, 1
    for (k, slot) in enumerate(layout.slots)
        for j in previous+1:slot-1
            gg[global_index] = gradient[j]
            global_index += 1
        end
        gz[layout.order[k]] = gradient[slot]
        previous = slot
    end
    for j in previous+1:length(gradient)
        gg[global_index] = gradient[j]
        global_index += 1
    end
    nothing
end
