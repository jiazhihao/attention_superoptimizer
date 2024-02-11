#pragma once

#include <memory>
#include "z3++.h"

namespace aso {
namespace search {

class AlgebraicPattern {
public:
  AlgebraicPattern() = default;
  virtual ~AlgebraicPattern() = default;

  virtual z3::expr to_z3(z3::context &c) const = 0;
  bool subpattern_to(AlgebraicPattern const &other) const;
  virtual std::string to_string() const = 0;
};

class Var : public AlgebraicPattern {
public:
  Var(std::string const &name);
  z3::expr to_z3(z3::context &c) const override;
  std::string to_string() const override;
  std::string name;
};

class Add : public AlgebraicPattern {
public:
  Add(std::shared_ptr<AlgebraicPattern> lhs,
      std::shared_ptr<AlgebraicPattern> rhs);
  z3::expr to_z3(z3::context &c) const override;
  std::string to_string() const override;
  std::shared_ptr<AlgebraicPattern> lhs, rhs;
};

class Mul : public AlgebraicPattern {
public:
  Mul(std::shared_ptr<AlgebraicPattern> lhs,
      std::shared_ptr<AlgebraicPattern> rhs);
  z3::expr to_z3(z3::context &c) const override;
  std::string to_string() const override;
  std::shared_ptr<AlgebraicPattern> lhs, rhs;
};

class Div : public AlgebraicPattern {
public:
  Div(std::shared_ptr<AlgebraicPattern> lhs,
      std::shared_ptr<AlgebraicPattern> rhs);
  z3::expr to_z3(z3::context &c) const override;
  std::string to_string() const override;
  std::shared_ptr<AlgebraicPattern> lhs, rhs;
};

class Exp : public AlgebraicPattern {
public:
  Exp(std::shared_ptr<AlgebraicPattern> exponent);
  z3::expr to_z3(z3::context &c) const override;
  std::string to_string() const override;
  std::shared_ptr<AlgebraicPattern> exponent;
};

class Red : public AlgebraicPattern {
public:
  Red(int k, std::shared_ptr<AlgebraicPattern> summand);
  z3::expr to_z3(z3::context &c) const override;
  std::string to_string() const override;
  int k;
  std::shared_ptr<AlgebraicPattern> summand;
};

} // namespace search
} // namespace aso