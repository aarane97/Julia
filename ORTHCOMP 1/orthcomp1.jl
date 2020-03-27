"""
z = orthcomp1(y, x)
Project `y` onto the orthogonal complement of `Span({x})`
In:
*`y` vector
*`x` nonzero vector of same length, both possibly very long
Out:*`z` vector of same length
For full credit, your solution should be computationally efficient.
"""
function orthcomp1(y, x)
    return y-((x'y)/(x'x))*x
end

y=[1;2;4]
x=[0;2;2]

orthcomp1(y, x)
