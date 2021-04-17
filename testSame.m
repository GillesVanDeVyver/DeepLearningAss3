function same = testSame(vec1,vec2, eps)
    T = (abs(vec1-vec2)/max(eps,abs(vec1)+abs(vec2))>eps);
    same = sum(T(:))==0;
end