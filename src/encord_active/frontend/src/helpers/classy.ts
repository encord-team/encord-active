export type ClassLike = string | null | undefined;
export type ClassRecord = Record<string, ClassLike | boolean>;
export type Classes = ClassLike | ClassLike[] | ClassRecord;

/**
 * Generates a `className` based on the specified values.
 * Supports multiple types, see example.
 * @example
 * const styles = { foo: 'foo-1g4k53' };
 * const globalClasses = ['global-1', 'global-2'];
 * const className = 'className';
 * const disabled = true;
 * const invalid = false;
 *
 * classy(styles.foo, globalClasses, className, { disabled, invalid });
 * // 'foo-1g4k53 global-1 global-2 className disabled'
 */
export const classy = (...classes: Classes[]): string =>
  classes
    .flat(10)
    .flatMap((cl) => (cl instanceof Object ? getTruthyKeys(cl) : cl))
    .filter(Boolean)
    .join(" ");

const getTruthyKeys = (obj: Record<string, unknown>): string[] =>
  Object.entries(obj)
    .filter(([, value]) => !!value)
    .map(([key]) => key);
