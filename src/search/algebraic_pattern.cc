#include "aso/search/algebraic_pattern.h"

namespace aso {
namespace search {

bool AlgebraicPattern::subpattern_to(AlgebraicPattern const &other) const {
  z3::context c;

  z3::sort P = c.uninterpreted_sort("Pattern");

  z3::func_decl add = z3::function("add", P, P, P);
  z3::func_decl mul = z3::function("mul", P, P, P);
  z3::func_decl div = z3::function("div", P, P, P);
  z3::func_decl exp = z3::function("exp", P, P);

  z3::func_decl subpattern = z3::partial_order(P, 0);

  z3::expr q = c.constant("q", P);
  z3::expr k = c.constant("k", P);
  z3::expr v = c.constant("v", P);

  z3::solver s(c);

  z3::params p(c);
  p.set("mbqi", true);
  p.set("timeout", 10u);
  s.set(p);

  s.add(q != k);
  s.add(q != v);
  s.add(k != v);

  z3::expr x = c.constant("x", P);
  z3::expr y = c.constant("y", P);
  z3::expr z = c.constant("z", P);

  s.add(forall(x, y, add(x, y) == add(y, x)));
  s.add(forall(x, y, mul(x, y) == mul(y, x)));
  s.add(forall(x, y, z, add(add(x, y), z) == add(x, add(y, z))));
  s.add(forall(x, y, z, mul(mul(x, y), z) == mul(x, mul(y, z))));
  s.add(forall(x, y, z, add(mul(x, z), mul(y, z)) == mul(add(x, y), z)));
  s.add(forall(x, y, z, add(div(x, z), div(y, z)) == div(add(x, y), z)));
  s.add(forall(x, y, z, mul(x, div(y, z)) == div(mul(x, y), z)));

  s.add(forall(x, y, subpattern(x, add(x, y))));
  s.add(forall(x, y, subpattern(x, mul(x, y))));
  s.add(forall(x, y, subpattern(x, div(x, y))));
  s.add(forall(x, y, subpattern(y, div(x, y))));
  s.add(forall(x, subpattern(x, exp(x))));

  s.add(to_z3(c) != other.to_z3(c));

  return s.check() == z3::unsat;
}

Var::Var(std::string const &name) : name(name) {}

z3::expr Var::to_z3(z3::context &c) const {
  z3::sort P = c.uninterpreted_sort("P");
  return c.constant(name.data(), P);
}

std::string Var::to_string() const {
  return name;
}

Add::Add(std::shared_ptr<AlgebraicPattern> lhs,
         std::shared_ptr<AlgebraicPattern> rhs)
    : lhs(lhs), rhs(rhs) {}

z3::expr Add::to_z3(z3::context &c) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl add = z3::function("add", P, P, P);
  return add(lhs->to_z3(c), rhs->to_z3(c));
}

std::string Add::to_string() const {
  return "(" + lhs->to_string() + "+" + rhs->to_string() + ")";
}

Mul::Mul(std::shared_ptr<AlgebraicPattern> lhs,
         std::shared_ptr<AlgebraicPattern> rhs)
    : lhs(lhs), rhs(rhs) {}

z3::expr Mul::to_z3(z3::context &c) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl mul = z3::function("mul", P, P, P);
  return mul(lhs->to_z3(c), rhs->to_z3(c));
}

std::string Mul::to_string() const {
  return "(" + lhs->to_string() + rhs->to_string() + ")";
}

Div::Div(std::shared_ptr<AlgebraicPattern> lhs,
         std::shared_ptr<AlgebraicPattern> rhs)
    : lhs(lhs), rhs(rhs) {}

z3::expr Div::to_z3(z3::context &c) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl div = z3::function("div", P, P, P);
  return div(lhs->to_z3(c), rhs->to_z3(c));
}

std::string Div::to_string() const {
  return "(" + lhs->to_string() + "/" + rhs->to_string() + ")";
}

Exp::Exp(std::shared_ptr<AlgebraicPattern> exponent) : exponent(exponent) {}

z3::expr Exp::to_z3(z3::context &c) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl exp = z3::function("exp", P, P);
  return exp(exponent->to_z3(c));
}

std::string Exp::to_string() const {
  return "e^" + exponent->to_string();
}

} // namespace search
} // namespace aso
