import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from './AuthContext';
import { AuthForm, Field } from './AuthForm';

export function RegisterPage() {
  const { register } = useAuth();
  const navigate = useNavigate();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');

  return (
    <AuthForm
      title="Create your account"
      subtitle="Start building your knowledge base"
      submitLabel="Create account"
      onSubmit={async () => {
        await register({
          email,
          password,
          // Omit rather than send an empty string — the API treats absent as
          // "no name given", and "" would be stored literally.
          ...(fullName.trim() ? { full_name: fullName.trim() } : {}),
        });
        navigate('/', { replace: true });
      }}
      footer={
        <>
          Already have an account?{' '}
          <Link to="/login" className="text-accent hover:opacity-80">
            Sign in
          </Link>
        </>
      }
    >
      <Field
        label="Full name"
        type="text"
        value={fullName}
        onChange={setFullName}
        autoComplete="name"
      />
      <Field
        label="Email"
        type="email"
        value={email}
        onChange={setEmail}
        autoComplete="email"
        required
      />
      <Field
        label="Password"
        type="password"
        value={password}
        onChange={setPassword}
        autoComplete="new-password"
        required
        minLength={8}
        hint="At least 8 characters. Length matters more than symbols."
      />
    </AuthForm>
  );
}
